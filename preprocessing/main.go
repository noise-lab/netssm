package main

import (
  "os"
  "fmt"
  "log"
  "flag"
  "bytes"
  "strings"
  "unicode"
  "crypto/sha256"
  "encoding/json"
  "encoding/csv"
  "encoding/binary"
  "path/filepath"

  "github.com/go-gota/gota/dataframe"
  "github.com/google/gopacket"
  "github.com/google/gopacket/layers"
  "github.com/google/gopacket/pcap"
)

type Capture struct {
  Data string
  Hash string
  Filename string
  OrigLen int
}

func hash(s string) string {
  h := sha256.New()
  h.Write([]byte(s))
  return fmt.Sprintf("%x", h.Sum(nil))
}

// Custom JSON Marshal function as detailed here: https://stackoverflow.com/a/28596225
func jsonMarshal(t interface{}) ([]byte, error) {
  buffer := &bytes.Buffer{}
  encoder := json.NewEncoder(buffer)
  encoder.SetEscapeHTML(false)
  err := encoder.Encode(t)
  return buffer.Bytes(), err
}

func truncateString(input string, n int) string {
  var result strings.Builder
  wordCount := 0
  inWord := false

  for _, r := range input {
    if unicode.IsSpace(r) {
      if inWord {
        wordCount++
        inWord = false
        if wordCount == n {
          break
        }
        result.WriteRune(' ')
      }
    } else {
      inWord = true
      result.WriteRune(r)
    }
  }
  return strings.TrimSpace(result.String())
}

func bytesToString(byteSlice []byte) string {
  strSlice := make([]string, len(byteSlice))
  for i, b := range byteSlice {
    strSlice[i] = fmt.Sprint(b)
  }
  return strings.Join(strSlice, " ")
}

func findIndex(df dataframe.DataFrame, column string, value string) (int, error) {
  col := df.Col(column)
  for i := 0; i < col.Len(); i++ {
    v := col.Val(i)
    if v == value {
      return i, nil
    }
  }
  return -1, fmt.Errorf("value %d not found in column %s", value, column)
}

func padTruncateSlice(index int, slice []byte, length int) []byte {
  padLength := length - len(slice)
  if padLength < 0 {
    fmt.Println(index, slice)
    fmt.Println(slice[:length])
    fmt.Println(len(slice[:length]))
    return slice[:length]
  }
  pad := make([]byte, padLength)
  return append(slice, pad...)
}

func getTCPLayer(packet gopacket.Packet) *layers.TCP {
  if packet == nil {
    return nil
  }
  for _, layer := range packet.Layers() {
    if tcpLayer, ok := layer.(*layers.TCP); ok {
      return tcpLayer
    }
  }
  return nil
}

func getTCPData(packet gopacket.Packet, pktdata []byte) []byte {
  transportFlag := false
  for _, layer := range packet.Layers() {
    // Skip Application or Error Layer
    _, isTransportLayer := layer.(gopacket.TransportLayer)
    _, isAppLayer := layer.(gopacket.ApplicationLayer)
    _, isErrorLayer := layer.(gopacket.ErrorLayer)
    if isAppLayer || isErrorLayer {
      continue
    } else {
      pktdata = append(pktdata, layer.LayerContents()...)
    }
    if isTransportLayer {
      transportFlag = true
    }
  }
  // Check if IHL is non-standard
  // if !transportFlag || len(pktdata) == 0 || pktdata[14] != 69 {
  if !transportFlag || len(pktdata) == 0 {
    return nil
  }
  return pktdata
}

func getUDPLayer(packet gopacket.Packet) *layers.UDP {
  if packet == nil {
    return nil
  }
  for _, layer := range packet.Layers() {
    if udpLayer, ok := layer.(*layers.UDP); ok {
      return udpLayer
    }
  }
  return nil
}

func getUDPData(packet gopacket.Packet, udp *layers.UDP, pktdata []byte) []byte {
  for _, layer := range packet.Layers() {
    // Extract only Link, Network for UDP logic below
    _, isLinkLayer := layer.(gopacket.LinkLayer)
    _, isNetworkLayer := layer.(gopacket.NetworkLayer)
    if isLinkLayer || isNetworkLayer {
      pktdata = append(pktdata, layer.LayerContents()...)
    }
  }

  // Extract UDP header
  header := make([]byte, 8) // UDP header is 8 bytes long
  binary.BigEndian.PutUint16(header[0:2], uint16(udp.SrcPort))
  binary.BigEndian.PutUint16(header[2:4], uint16(udp.DstPort))
  binary.BigEndian.PutUint16(header[4:6], udp.Length)
  binary.BigEndian.PutUint16(header[6:8], udp.Checksum)
  pktdata = append(pktdata, header...)
  // Append the first 12 bytes of the payload, to capture any significant UDP application layer headers (e.g., RTP, DNS)
  if len(udp.Payload) >= 12 {
    pktdata = append(pktdata, udp.Payload[:12]...)
  } else {
    pktdata = append(pktdata, udp.Payload[:len(udp.Payload) - 1]...)
  }
  return pktdata
}

func getPcapData(handle *pcap.Handle) ([]string, int) {
  var pcap []string
  var pktcount int

  packetSource := gopacket.NewPacketSource(handle, handle.LinkType())
  for packet := range packetSource.Packets() {
    tcpLayer := getTCPLayer(packet)
    udpLayer := getUDPLayer(packet)
    if tcpLayer == nil && udpLayer == nil {
      continue
    }

    // Grab packet data, including padded options, but no payload
    var pktdata []byte
    if tcpLayer != nil {
      pktdata = getTCPData(packet, pktdata)
      if pktdata == nil {
        return nil, -1
      }
    } else if udpLayer != nil {
      pktdata = getUDPData(packet, udpLayer, pktdata)
      if pktdata == nil {
        return nil, -1
      }
    } else {
      return nil, -1
    }
    // Pad to length 128 (eth:14, max[ipv4 (20), ipv6 (40)], max[tcp (60), udp (20)], udp app. layer hdr:12 == 126, +2 for nice value and model dim. niceties)
    if len(pktdata) != 128 {
      pktdata = padTruncateSlice(pktcount, pktdata, 128)
      if pktdata == nil {
        return nil, -1
      }
    }
    strData := bytesToString(pktdata)
    pcap = append(pcap, strData)
    pktcount++
  }
  if pktcount <= 1 {
    return nil, -1
  }
  return pcap, pktcount
}

func tagPcapData(labels dataframe.DataFrame, pcapfile string, input string, label string) string {
  var raw_label string
  if label != "" {
    raw_label = label
  } else {
    if strings.Contains(pcapfile, "/") {
      pcapfile = filepath.Base(pcapfile)
    }
    index, err := findIndex(labels, "File", pcapfile)
    if err != nil {
      log.Println(err)
    }
    raw_label = labels.Elem(index, 1).String()
  }
  l := "<|" + raw_label + "|> "
  input = l + input
  return input
}

func main() {
  var pcaps []string
  var df dataframe.DataFrame
  pcap_num := 1

  srcPcapsPtr := flag.String("in-dir", "./", "Directory with PCAPs")
  srcPcapsCSVPtr := flag.String("in-csv", "./", "CSV file with PCAP paths")
  outfilePtr := flag.String("out", "./", "Output JSONL dataset name")
  labelsFilePtr := flag.String("label-csv", "./", "CSV mapping pcaps in `in-dir` to their corresp. label")
  labelPtr := flag.String("label", "", "Blanket label to use, if all PCAPs are of same service/type")
  truncatePtr := flag.Int("truncate", -1, "OPTIONAL -- Length to truncate all samples to")

  flag.Parse()

  // CL args checking
  if *srcPcapsPtr == "./" && *srcPcapsCSVPtr == "./" && *labelsFilePtr == "./" && *labelPtr == "" {
    flag.Usage()
    os.Exit(0)
  }
  if *outfilePtr == "./" {
    log.Fatalf("Error: Must provide a name for output JSONL file")
  }
  if *srcPcapsPtr == "./" && *srcPcapsCSVPtr == "./" {
    log.Fatalf("Error: Must provide either `in-dir` or `-in-csv`")
  }
  if *labelsFilePtr == "./" && *labelPtr== "./" {
    log.Fatalf("Error: Must provide either `label-csv` or `label`")
  }
  if *srcPcapsPtr != "./" && *srcPcapsCSVPtr != "./" {
  	log.Fatalf("Error: Cannot provide both -in-dir and -in-csv options")
  }
  if *labelsFilePtr != "./" && *labelPtr != "" {
    log.Fatalf("Error: Cannot provide both -labels and -label options")
  }

  // Get all PCAP files from the provided directory
  if *srcPcapsPtr != "./" {
    filepath.Walk(*srcPcapsPtr, func(path string, info os.FileInfo, err error) error {
      if err != nil {
        if os.IsPermission(err) {
          log.Printf("Skipping directory due to permission denied: %s", path)
          return filepath.SkipDir
        }
        log.Fatalf(err.Error())
      }
      if info.Mode().IsRegular() && (filepath.Ext(info.Name()) == ".pcap" || filepath.Ext(info.Name()) == ".pcapng") {
        pcaps = append(pcaps, path)
      }
      return nil
    })
  // Or from CSV file of paths to PCAPs
  } else if *srcPcapsCSVPtr != "./" {
	csvFile, err := os.Open(*srcPcapsCSVPtr)
	if err != nil {
	  log.Fatalf("Error opening CSV file: %v\n", err)
	}
	defer csvFile.Close()
	reader := csv.NewReader(csvFile)
	records, err := reader.ReadAll()
	if err != nil {
	  log.Fatalf("Error reading CSV file: %v\n", err)
	}
	if len(records) < 2 {
	  log.Println("No data found in CSV file.")
	  return
	}
	for _, row := range records {
      if len(row) > 0 {
        pcaps = append(pcaps, row[0])
      }
	}
  }

  // Create labels dataframe for lookup
  if *labelsFilePtr != "./" {
    l_file, err := os.Open(*labelsFilePtr)
    defer l_file.Close()
    if err != nil {
      log.Fatal(err)
    }
    df = dataframe.ReadCSV(l_file)
  }

  // Create output JSON file
  out, err := os.OpenFile(*outfilePtr, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
  if err != nil {
    log.Fatal(err)
  }

  // Build dataset
  for _, pf := range pcaps {
    handle, err := pcap.OpenOffline(pf)
    if err != nil {
      log.Println("Unable to open " + pf + " as PCAP. Continuing...")
      log.Println(len(pcaps) - pcap_num, "PCAP files remaining")
      pcap_num = pcap_num + 1
      continue
    }

    pcapData, pktcount := getPcapData(handle)
    if len(pcapData) <= 1 || pcapData == nil || pktcount == -1{
      log.Println("Got error when parsing PCAP " + pf + ", or PCAP has len. of 1. Continuing...")
      log.Println(len(pcaps) - pcap_num, "PCAP files remaining")
      pcap_num = pcap_num + 1
      continue
    }
    pcapInput := strings.Join(pcapData, " <|pkt|> ")
    pcapInput = tagPcapData(df, pf, pcapInput, *labelPtr)

    if *truncatePtr != -1 {
      pcapInput = truncateString(pcapInput, *truncatePtr)
    }
    p := Capture{pcapInput, hash(pf), pf, pktcount}
    b, err := jsonMarshal(p)
    if err != nil {
      log.Fatal(err)
    }

    out.WriteString(string(b))

    log.Println("Finished parsing PCAP file: ", pf)
    log.Println(len(pcaps) - pcap_num, "PCAP files remaining")
    pcap_num = pcap_num + 1
    handle.Close()
  }
  out.Close()
}
