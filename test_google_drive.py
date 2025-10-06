#!/usr/bin/env python3
"""
Test Google Drive file accessibility
"""
import requests

def test_google_drive_file():
    """Test if the Google Drive file is accessible with virus scan bypass"""
    file_id = "1Zenpa8iO8fWWv2zNBrlUDIpY0kc3yDRj"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    print(f"Testing Google Drive file: {file_id}")
    print(f"URL: {url}")
    
    try:
        # First request to get the virus scan warning page
        response = requests.get(url, timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Content-Type: {response.headers.get('content-type', 'unknown')}")
        
        # Check if we got the virus scan warning page
        if "virus scan warning" in response.text.lower():
            print("Detected virus scan warning, attempting to bypass...")
            
            # Extract the confirmation token from the page
            import re
            confirm_pattern = r'name="confirm"\s+value="([^"]+)"'
            match = re.search(confirm_pattern, response.text)
            
            if match:
                confirm_token = match.group(1)
                print(f"Found confirmation token: {confirm_token}")
                
                # Make second request with confirmation token
                confirm_url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={confirm_token}"
                print(f"Confirmation URL: {confirm_url}")
                
                response = requests.get(confirm_url, timeout=10)
                print(f"Confirmation response status: {response.status_code}")
                print(f"Confirmation Content-Type: {response.headers.get('content-type', 'unknown')}")
                print(f"Confirmation Content-Length: {response.headers.get('content-length', 'unknown')}")
                
                if response.status_code == 200:
                    # Check if it's actually a file now
                    content = response.text[:200] if len(response.text) > 200 else response.text
                    if "random_forest" in content.lower() or "pkl" in content.lower() or len(response.content) > 1000000:
                        print("SUCCESS: Appears to be a pickle file after bypass")
                    else:
                        print("WARNING: Content still doesn't look like a pickle file")
                        print(f"First 200 chars: {content}")
                else:
                    print(f"ERROR: Confirmation request failed with HTTP {response.status_code}")
            else:
                print("ERROR: Could not find confirmation token in virus scan page")
        else:
            print("No virus scan warning detected")
            
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    test_google_drive_file()
