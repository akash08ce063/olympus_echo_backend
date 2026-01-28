# Pranthora-LiveKit Bridge Test Setup Steps

This document provides step-by-step instructions for setting up and running the Pranthora-LiveKit bridge test.

## Prerequisites

1. **Python Dependencies**
   - Install required packages:
     ```bash
     cd olympus_echo_backend
     pip install -r requirements.txt
     ```

2. **Configuration**
   - Ensure `config.json` contains LiveKit credentials:
     ```json
     {
       "livekit": {
         "url": "wss://jill-80u1rdyp.livekit.cloud",
         "api_key": "APIG236hF2mb5hp",
         "api_secret": "CU743exSdHD1JeIJf0UzUdTjClVrYbl9tuUefvttfBeb"
       }
     }
     ```

3. **Pranthora Agent**
   - Ensure the Pranthora agent is accessible at:
     `wss://api.pranthora.com/api/call/web-media-stream?agent_id=e1143125-5a7f-4955-8f2c-01df1c604343`

## LiveKit Webhook Setup (OPTIONAL)

**⚠️ IMPORTANT: Webhooks are OPTIONAL for this test!**

The test will work perfectly fine without webhooks. Webhooks are only needed if you want to receive event notifications (like when participants join/leave, rooms start/finish, etc.) to an external service.

### If You Want to Set Up Webhooks (Optional)

If you have a webhook endpoint URL and want to receive LiveKit events:

#### Step 1: Access LiveKit Dashboard
1. Go to your LiveKit Cloud dashboard
2. Navigate to your project: `jill-80u1rdyp.livekit.cloud`

#### Step 2: Create Webhook
1. Go to **Settings** → **Webhooks** (or **Integrations** → **Webhooks`)
2. Click **Add Webhook** or **Create Webhook** button
3. The webhook dialog will appear with these fields:
   - **Name**: Enter a name for your webhook (e.g., "My Webhook")
   - **URL**: Enter your webhook endpoint URL (e.g., `https://my.domain/webhook`)
     - **Where to get this?**: You need to have a server/endpoint that can receive POST requests from LiveKit
     - **If you don't have one**: You can skip webhook setup entirely - the test will work without it!
   - **Signing API key**: Select an API key from the dropdown (or use your existing API key)
     - This is used to sign webhook requests so you can verify they come from LiveKit
     - Click "Learn more" link for details

#### Step 3: Save Webhook
- Click **Create** button
- The webhook will now receive all LiveKit events automatically

### If You Don't Have a Webhook Endpoint

**You can completely skip webhook setup!** The test doesn't require webhooks to function. Webhooks are only for receiving event notifications to an external service.

**When would you need webhooks?**
- If you have a server/application that needs to be notified when LiveKit events occur (e.g., participant joins, room starts, etc.)
- If you're building a production system that needs to track LiveKit activity
- If you want to integrate LiveKit events with other services

**For this test:** You don't need webhooks at all. The test directly connects to LiveKit using the API and WebSocket, so it doesn't rely on webhook notifications.

Simply proceed to the "Running the Test" section below.

## Running the Test

### Step 1: Navigate to Test Directory
```bash
cd olympus_echo_backend/tests
```

### Step 2: Run the Test
```bash
python test_pranthora_livekit_bridge.py
```

Or make it executable:
```bash
chmod +x test_pranthora_livekit_bridge.py
./test_pranthora_livekit_bridge.py
```

### Step 3: Monitor Output
The test will:
1. ✅ Create or get a LiveKit room
2. ✅ Generate a participant token
3. ✅ Connect to LiveKit room
4. ✅ Print the LiveKit WebSocket URL (save this!)
5. ✅ Connect to Pranthora agent
6. ✅ Bridge audio between the two
7. ✅ Run for 60 seconds (configurable via `MAX_DURATION_SEC`)

### Step 4: Check the Output
Look for these key outputs:
- **LiveKit Participant WebSocket URL**: This is the URL you can use to connect another agent/participant
- **Connection status**: Both connections should show "✅ Connected"
- **Audio bridge status**: Should show "🎵 Audio bridge is active!"

## What the Test Does

1. **Creates LiveKit Room**: If the room doesn't exist, it creates one. Otherwise, it uses the existing room.

2. **Connects as Participant**: The test script connects to the LiveKit room as a participant (not as an agent service).

3. **Bridges Audio**:
   - Audio from Pranthora → LiveKit room
   - Audio from LiveKit room → Pranthora

4. **Prints WebSocket URL**: The LiveKit participant WebSocket URL is printed, which you can use to:
   - Connect another agent to the same room
   - Connect a client application
   - Test with multiple participants

## Connecting Another Agent to the Room

To connect a second agent (Agent 2) to the LiveKit room:

1. **Get the WebSocket URL** from the test output
2. **Use LiveKit SDK** to connect:
   ```python
   from livekit import rtc
   
   room = rtc.Room()
   await room.connect(ws_url, token)
   ```

3. **Or use the LiveKit Agents framework**:
   - Create an agent using LiveKit Agents SDK
   - Configure it to join the room using the room name and token

## Troubleshooting

### Issue: "Error creating LiveKit room"
- **Solution**: Check your LiveKit API credentials in `config.json`
- Verify the LiveKit URL is correct

### Issue: "Error connecting to Pranthora agent"
- **Solution**: 
  - Check if the Pranthora agent ID is correct
  - Verify network connectivity to `api.pranthora.com`
  - Check if the agent is active and accessible

### Issue: "No audio being bridged"
- **Solution**:
  - Check that both connections are established
  - Verify audio format compatibility (8kHz, mono, 16-bit PCM)
  - Check queue sizes and processing delays

### Issue: "Webhook not working"
- **Note**: Webhooks are optional and not required for the test to work
- **If you set up webhooks**: 
  - Verify webhook is configured in LiveKit dashboard
  - Check webhook endpoint is accessible and can receive POST requests
  - Review webhook logs in LiveKit dashboard
  - Verify the signing API key matches

## Test Duration

The test runs for **60 seconds** by default. You can modify `MAX_DURATION_SEC` in the test file to change this.

## Cleanup

The test automatically:
- Disconnects from LiveKit room
- Closes Pranthora WebSocket connection
- Cleans up resources

Note: The LiveKit room will remain active until it times out (5 minutes of inactivity by default).

## Next Steps

After running the test successfully:
1. Use the printed WebSocket URL to connect additional agents
2. Monitor the conversation between agents
3. Adjust audio settings if needed
4. Extend the test for longer conversations

## Additional Resources

- [LiveKit Documentation](https://docs.livekit.io/)
- [LiveKit Python SDK](https://github.com/livekit/python-sdk)
- [LiveKit Agents](https://docs.livekit.io/agents/overview/)
