import os
import subprocess
import sys
import pandas as pd

def run_command(command):
    """Executes a shell command and prints its output."""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    print(result.stdout)

def get_latest_report(output_dir, base_name):
    """Finds the latest MMT report file based on the timestamped prefix."""
    files = [f for f in os.listdir(output_dir) if f.endswith(f"{base_name}.pcap.csv")]
    if not files:
        print("Error: No report file found.")
        sys.exit(1)
    latest_report = max(files, key=lambda f: os.path.getctime(os.path.join(output_dir, f)))
    return os.path.join(output_dir, latest_report)

def process_pcap(pcap_file, is_malicious):
    """
    Runs the full workflow from pcap processing to CSV conversion.
    """
    base_name = os.path.splitext(os.path.basename(pcap_file))[0]
    output_dir = f"/tmp/{base_name}-reports/"
    output_pkl = f"/tmp/{base_name}.pkl"
    output_csv = f"/tmp/{base_name}.csv"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Step 1: Run mmt-probe to generate MMT reports
    mmt_command = f"mmt-probe -c mmt-probe.conf -t {pcap_file} -X \"file-output.output-dir={output_dir}\" -X \"file-output.output-file={base_name}.pcap.csv\""
    run_command(mmt_command)

    # Step 2: Find the latest generated report file
    report_csv = get_latest_report(output_dir, base_name)

    # Step 3: Extract features from the generated report
    feature_extraction_command = f"python3 utils/trafficToFeature.py {report_csv} {output_pkl} {is_malicious}"
    run_command(feature_extraction_command)

    # Step 4: Convert the pickle file to CSV
    convert_pkl_to_csv(output_pkl, output_csv)

def convert_pkl_to_csv(input_pkl, output_csv):
    """Converts a pickle (.pkl) file to a CSV file."""
    try:
        print(f"Loading pickle file: {input_pkl}")
        df = pd.read_pickle(input_pkl)

        print(f"Checking data integrity ({len(df)} rows before cleaning)")
        df = df[df.notnull().all(axis=1)]  # Remove rows with NaN values
        df = df.replace([float('inf'), float('-inf')], 0)  # Replace infinities
        print(f"Data cleaned ({len(df)} rows after cleaning)")

        df.to_csv(output_csv, index=False)
        print(f"Successfully saved CSV: {output_csv}")
    except Exception as e:
        print(f"Error processing file {input_pkl}: {e}")
        sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python process_pcap.py <pcap_file> <is_malicious (True/False)>")
        sys.exit(1)

    pcap_file = sys.argv[1]
    is_malicious = sys.argv[2].lower() == "true"
    process_pcap(pcap_file, is_malicious)
