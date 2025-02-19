# fingerprint.py
import acoustid
import base64

def generate_fingerprint(file_path):
    """
    Generates a robust audio fingerprint using the Chromaprint algorithm.
    Chromaprint is designed to be resilient to changes in key, BPM, and duration.
    
    Returns:
        A Base64-encoded string fingerprint that can be used to check audio uniqueness.
    """
    try:
        # acoustid.fingerprint_file returns (duration, fingerprint)
        duration, fp = acoustid.fingerprint_file(file_path)
        # If fp is bytes, encode it with base64
        if isinstance(fp, bytes):
            fp = base64.b64encode(fp).decode('utf-8')
        return fp
    except Exception as e:
        raise RuntimeError(f"Error generating fingerprint: {e}")
