import os
import re
import subprocess
import xml.etree.ElementTree as ET


def load_qpme(qpme_file):
    """
    Load a QPME XML (.qpe) file and return the parsed ElementTree and root element.
    """
    tree = ET.parse(qpme_file)
    root = tree.getroot()
    return tree, root


def save_qpme(tree, output_file):
    """
    Save the modified XML tree to a file.
    """
    tree.write(output_file, encoding="UTF-8", xml_declaration=True)
    print(f"Saved modified QPME model to '{output_file}'")


def modify_qpme(tree, root, element):
    """
    Modify specific <place> elements in the QPME model.
    Example: change mean and stdDev of all 'Normal' distribution color-refs
    """
    namespace = {'xsi': 'http://www.w3.org/2001/XMLSchema-instance'}  # QPME often uses xsi namespace
    element_name = element['element']
    mean = element['mean']
    std = element['std']

    # Find all <place> elements
    for place in root.findall(".//place"):
        place_name = place.attrib.get('name', '')
        if place_name.startswith(element_name):
            color_refs = place.find("color-refs")
            if color_refs is not None:
                for color_ref in color_refs.findall("color-ref"):
                    if color_ref.attrib.get('distribution-function') == "Normal":
                        if "Workload" in place_name:
                            frequency = element['frequency']
                            mean = f"{60 / float(frequency):.4f}"
                        old_mean = color_ref.attrib.get('mean')
                        old_std = color_ref.attrib.get('stdDev')
                        color_ref.set('mean', mean)  # Change mean to 5 (example)
                        color_ref.set('stdDev', std)  # Change stdDev to 2 (example)
                        print(f"Updated {place_name}: mean {old_mean} → {mean}, stdDev {old_std} → {std}")
                    if color_ref.attrib.get('distribution-function') == "Exponential":
                        if "Workload" in place_name:
                            frequency = element['frequency']
                            new_lambda = f"{float(frequency)/60:.4f}"
                        else:
                            new_lambda = f"{1 / float(mean):.4f}"
                        old_lambda = color_ref.attrib.get('lambda')
                        color_ref.set('lambda', new_lambda)  # Change mean to 5 (example)
                        print(f"Updated {place_name}: lambda {old_lambda} → {new_lambda}")
    return tree


'''
    </place>
    <place id="_1638358621573" departure-discipline="NORMAL" xsi:type="queueing-place" name="Workload" queue-ref="_1638358621663">
      <meta-attributes>
        <meta-attribute xsi:type="location-attribute" location-x="176" location-y="433"/>
        <meta-attribute xsi:type="simqpn-place-configuration" id="_1699036119368" statsLevel="3" configuration-name="config"/>
      </meta-attributes>
      <color-refs>
        <color-ref maximum-capacity="0" id="_1638358621574" xsi:type="queueing-color-reference" ranking="0" priority="0" color-id="_1638358621575" initial-population="1" distribution-function="Normal" stdDev="1" mean="10">
          <meta-attributes>
            <meta-attribute xsi:type="simqpn-batch-means-queueing-color-configuration" id="_1699036119440" signLev="0.05" reqAbsPrc="50" reqRelPrc="0.05" batchSize="200" minBatches="60" numBMeansCorlTested="50" bucketSize="100.0" maxBuckets="1000" queueSignLev="0.05" queueReqAbsPrc="50" queueReqRelPrc="0.05" queueBatchSize="200" queueMinBatches="60" queueNumBMeansCorlTested="50" queueBucketSize="100.0" queueMaxBuckets="1000" configuration-name="config"/>
          </meta-attributes>
        </color-ref>
      </color-refs>
    </place>
    <place id="_1638277914542" departure-discipline="NORMAL" xsi:type="queueing-place" name="W/index" queue-ref="_1638277914538">
      <meta-attributes>
        <meta-attribute xsi:type="location-attribute" location-x="509" location-y="82"/>
        <meta-attribute xsi:type="simqpn-place-configuration" id="_1699036119349" statsLevel="3" configuration-name="config"/>
      </meta-attributes>
      <color-refs>
        <color-ref initial-population="0" maximum-capacity="0" id="_1638350503823" xsi:type="queueing-color-reference" ranking="0" priority="0" color-id="_1638350503813" distribution-function="Normal" stdDev="1" mean="3">
          <meta-attributes>
            <meta-attribute xsi:type="simqpn-batch-means-queueing-color-configuration" id="_1699036119401" signLev="0.05" reqAbsPrc="50" reqRelPrc="0.05" batchSize="200" minBatches="60" numBMeansCorlTested="50" bucketSize="100.0" maxBuckets="1000" queueSignLev="0.05" queueReqAbsPrc="50" queueReqRelPrc="0.05" queueBatchSize="200" queueMinBatches="60" queueNumBMeansCorlTested="50" queueBucketSize="100.0" queueMaxBuckets="1000" configuration-name="config"/>
          </meta-attributes>
        </color-ref>
      </color-refs>
    </place>
'''


def run_qpme(_config, _output_result_file, _qpme_file, _exec_path):
    qpme_command = [
        _exec_path,
        "-r", _config,
        "-o", _output_result_file,
        _qpme_file
    ]

    # Set Java options for more heap
    env = os.environ.copy()
    env["JAVA_OPTS"] = "-Xmx4g -XX:-UseGCOverheadLimit"

    try:
        result = subprocess.run(
            qpme_command,
            capture_output=True,
            text=True,
            check=False,  # Avoid exception on non-zero exit
            cwd="/Users/yuanjiexia/Applications/qpme.app/Contents/Eclipse"
        )
        print(f"[QPME] Command executed with exit code {result.returncode}")
        if result.stdout:
            print("[QPME STDOUT]")
            print(result.stdout)
        if result.stderr:
            print("[QPME STDERR]")
            print(result.stderr)

        return result.returncode, result.stdout, result.stderr

    except Exception as e:
        print(f"[QPME] Failed to run simulation: {e}")
        return -1, "", str(e)


def extract_ranked_queue_utils(qpme_analysis_file):
    """
    Extracts all queueUtilQPl values and returns elements ranked by utilization.

    :param qpme_analysis_file: Path to QPME XML analysis output
    :return: List of tuples (element_name, utilization_value) sorted by utilization descending
    """
    try:
        tree = ET.parse(qpme_analysis_file)
        root = tree.getroot()

        util_list = []
        meanST_list = []

        for observed in root.findall(".//observed-element"):
            if observed.attrib.get("type") == "qplace:queue":
                element_name = observed.attrib.get("name")
                if element_name == "Workload":
                    continue  # Skip workload element
                for metric in observed.findall("metric"):
                    if metric.attrib.get("type") == "queueUtilQPl":
                        util = float(metric.attrib["value"])
                        util_list.append((element_name, util))
                        print(f"[QPME] {element_name}: queueUtilQPl = {util:.6f}")

                for color in observed.findall("color"):
                    for metric in color.findall("metric"):
                        if metric.attrib.get("type") == "meanST":
                            meanST = float(metric.attrib["value"])
                            meanST_list.append((element_name, meanST))
                            print(f"[QPME] {element_name}: meanST = {meanST:.6f}")
        # Sort by utilization descending
        util_list.sort(key=lambda x: x[1], reverse=True)
        meanST_list.sort(key=lambda x: x[1], reverse=True)

        print("\n[QPME] Ranked Utilization:")
        for rank, (el, val) in enumerate(util_list, start=1):
            print(f"{rank}. {el} = {val:.6f}")

        print("\n[QPME] Ranked meanSt:")
        for rank, (el, val) in enumerate(meanST_list, start=1):
            print(f"{rank}. {el} = {val:.6f}")

        return util_list, meanST_list

    except Exception as e:
        print(f"[ERROR] Failed to parse analysis file: {e}")
        return []


def qpme_running(_change_element_set, _qpme_file_location="./qpme/teastore_model.qpe"):
    _config_name = "config"
    cwd = os.getcwd()

    pattern = re.compile(r"qpme_output_(\d+)\.qpe")
    max_index = -1
    for filename in os.listdir(f"{cwd}/qpme"):
        match = pattern.match(filename)
        if match:
            index = int(match.group(1))
            max_index = max(max_index, index)

    index = max_index + 1
    _modified_file_name = f"{cwd}/qpme/qpme_output_{index}.qpe"

    _exec_path = "/Users/yuanjiexia/Applications/qpme.app/Contents/Eclipse/SimQPN.sh"
    _qpme_analysis_file_name = f"{cwd}/qpme/analysis_result_{index}"
    tree, root = load_qpme(_qpme_file_location)
    for item in _change_element_set:
        modified_tree = modify_qpme(tree, root, item)
    save_qpme(modified_tree, _modified_file_name)
    run_qpme(_config_name, _qpme_analysis_file_name, _modified_file_name, _exec_path)
    _ranked_list, _mean_st_list = extract_ranked_queue_utils(_qpme_analysis_file_name)
    return _ranked_list, _mean_st_list


if __name__ == '__main__':
    _change_element_set_list = [
        {
            "element": "Workload",
            "mean": "10",
            "std": "0.5"
        },
        {
            "element": "P/categories_login",
            "mean": "7",
            "std": "0.5"
        }
    ]
    qpme_running(_change_element_set_list)