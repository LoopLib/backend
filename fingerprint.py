# fingerprint.py
import json
import subprocess

def generate_fingerprint(file_path):
    """
    Generates an audio fingerprint using the Chromaprint command-line tool (fpcalc).
    
    :param file_path: Path to the audio file.
    :return: The fingerprint as a string or None if an error occurs.
    """
    try:
        result = subprocess.run(
            ['fpcalc', '-json', file_path],
            capture_output=True,
            text=True,
            check=True
        )
        fingerprint_data = json.loads(result.stdout)
        return fingerprint_data.get("fingerprint")
    except Exception as e:
        print("Error generating fingerprint:", e)
        return None
