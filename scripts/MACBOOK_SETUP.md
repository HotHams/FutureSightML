# MacBook Certificate Setup — Claude Handoff

Run this on the MacBook with Claude Code. Copy everything below the line into Claude.

---

## Task

I need you to set up Apple Developer ID code signing for my GitHub repo `HotHams/FutureSightML`. Do everything step by step, confirming with me before running destructive commands. Here's what needs to happen:

### Step 1: Generate a Certificate Signing Request (CSR)

Run this in bash (ask me for my Apple ID email and full name first):

```bash
openssl req -new -newkey rsa:2048 -nodes \
  -keyout ~/Desktop/devid.key \
  -out ~/Desktop/CertificateSigningRequest.certSigningRequest \
  -subj "/emailAddress=MY_EMAIL/CN=MY_NAME"
```

Then tell me to:
1. Go to https://developer.apple.com/account/resources/certificates/add
2. Select "Developer ID Application"
3. Select "G2 sub-CA"
4. Upload `~/Desktop/CertificateSigningRequest.certSigningRequest`
5. Download the resulting `.cer` file to `~/Desktop/`

Ask me to confirm when I've downloaded it.

### Step 2: Import the certificate and create a .p12

```bash
# Find the .cer file
CER_FILE=$(ls ~/Desktop/developerID_application*.cer 2>/dev/null | head -1)
# If the name is different, ask me what it's called

# Import into keychain
security import "$CER_FILE" -k ~/Library/Keychains/login.keychain-db

# Combine private key + cert into a .p12 (will ask me for a password)
# First find the cert's common name
security find-identity -v -p codesigning | grep "Developer ID Application"
```

Then export as .p12:
```bash
# Create p12 from the private key and downloaded cert
openssl x509 -inform DER -in "$CER_FILE" -out ~/Desktop/devid.pem
openssl pkcs12 -export \
  -inkey ~/Desktop/devid.key \
  -in ~/Desktop/devid.pem \
  -out ~/Desktop/cert.p12
# It will prompt for an export password — I need to remember this
```

### Step 3: Base64 encode the .p12

```bash
base64 -i ~/Desktop/cert.p12 -o ~/Desktop/cert.b64
cat ~/Desktop/cert.b64
```

Save this output — it's the `APPLE_CERTIFICATE` secret value.

### Step 4: Generate an app-specific password

Tell me to:
1. Go to https://account.apple.com/ (or appleid.apple.com)
2. Sign-In and Security → App-Specific Passwords
3. Click "+" or "Generate" → name it "GitHub CI"
4. Copy the generated password

### Step 5: Get my Team ID

```bash
# Try to find it from existing provisioning profiles or certs
security find-certificate -c "Developer ID Application" -p ~/Library/Keychains/login.keychain-db | openssl x509 -noout -subject
```

Or tell me to go to https://developer.apple.com/account → Membership Details → Team ID (10-character string).

### Step 6: Set GitHub secrets

Use `gh` CLI to set all 5 secrets (ask me if gh is installed, install with `brew install gh` if not):

```bash
# Login to GitHub first if needed
gh auth status || gh auth login

# Set secrets (will prompt for values)
gh secret set APPLE_CERTIFICATE --repo HotHams/FutureSightML < ~/Desktop/cert.b64
```

Then ask me for each value and set:
```bash
gh secret set APPLE_CERTIFICATE_PASSWORD --repo HotHams/FutureSightML
gh secret set APPLE_ID --repo HotHams/FutureSightML
gh secret set APPLE_ID_PASSWORD --repo HotHams/FutureSightML
gh secret set APPLE_TEAM_ID --repo HotHams/FutureSightML
```

### Step 7: Clean up sensitive files

```bash
rm ~/Desktop/devid.key ~/Desktop/devid.pem ~/Desktop/cert.p12 ~/Desktop/cert.b64
# Keep the .cer as backup if desired
```

### Step 8: Verify

```bash
gh secret list --repo HotHams/FutureSightML
```

Should show all 5 secrets. Tell me we're done and that I can push a `v*` tag from my Windows machine to trigger a fully signed macOS build.
