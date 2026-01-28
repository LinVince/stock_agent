from pyngrok import ngrok

# paste your actual authtoken as a string
ngrok.set_auth_token("2SHNoe3P8K0M9rQBHoyVfW8Erg6_4v9hML9SNNRfQjygN1soZ")
public_url = ngrok.connect(5000)
print(public_url)
