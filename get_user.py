import json, requests, jwt, base64, os


def get_user(request):
  if 'ROUTE' in os.environ and os.environ['ROUTE'] == 'api/case':
        # Get the key id from JWT headers (the kid field)
        encoded_jwt = request.headers['x-amzn-oidc-data']
        jwt_headers = encoded_jwt.split('.')[0]
        decoded_jwt_headers = base64.b64decode(jwt_headers)
        decoded_jwt_headers = decoded_jwt_headers.decode("utf-8")
        decoded_json = json.loads(decoded_jwt_headers)
        kid = decoded_json['kid']

        # Get the public key from regional endpoint
        url = 'https://public-keys.auth.elb.us-east-1.amazonaws.com/' + kid
        req = requests.get(url)
        pub_key = req.text

        # Get the payload
        payload = jwt.decode(encoded_jwt, pub_key, algorithms=['ES256'])
        return {
            'name': payload["name"],
            'email': payload['preferred_username']
        }
  else:
      # For local env
      return {
        'name': 'Test User',
        'email': 'test@fake.org'
    }