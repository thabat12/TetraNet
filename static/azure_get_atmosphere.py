import urllib.request
import json
import os
import ssl


def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

def run_ANN():
    allowSelfSignedHttps(True)  # this line is needed if you use self-signed certificate in your scoring service.

    data = {
        "data":
            [
                {
                    'Column1': "0",
                    'Column2': "0",
                    'Column3': "0",
                    'Column4': "0",
                    'Column5': "0",
                    'Column6': "0",
                    'Column7': "0",
                    'Column8': "1.75",
                    'Column9': "4.61",
                    'Column10': "6.83",
                    'Column11': "8.71",
                    'Column12': "10.13",
                },
            ],
    }

    body = str.encode(json.dumps(data))

    url = 'http://2d752c31-8eb1-48fe-94eb-4f0e306f1c56.eastus.azurecontainer.io/score'
    api_key = 'WJp4MQLmjaDDXCG5uYLJm6tz1OdNn2E1'  # Replace this with the API key for the web service
    headers = {'Content-Type': 'application/json', 'Authorization': ('Bearer ' + api_key)}

    req = urllib.request.Request(url, body, headers)

    try:
        response = urllib.request.urlopen(req)

        result = response.read()
        print(result)
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))

        # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
        print(error.info())
        print(json.loads(error.read().decode("utf8", 'ignore')))
