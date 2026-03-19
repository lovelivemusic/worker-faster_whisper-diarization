# RunPod Serverless Billing

## Understanding Serverless Costs

When using RunPod Serverless, you're billed for **total worker uptime**, not just active processing time.

### The Idle Timeout Gotcha

With a 600 second (10 minute) idle timeout:

1. You send a transcription request
2. Worker spins up, processes for ~40 seconds
3. Worker stays **warm and billing** for 600 more seconds waiting for another request
4. Worker finally shuts down

**Example:**
- Job processing time: 40 seconds
- Idle timeout: 600 seconds
- **Total billed time: 640 seconds**

At ~$0.00034/sec GPU rate:
- App shows: `GPU: 40.1s ($0.0136)` (active time only)
- Actual cost: 640s × $0.00034 = **~$0.22**

### Recommended Settings

| Use Case | Idle Timeout | Trade-off |
|----------|--------------|-----------|
| Infrequent use | 30-60 seconds | Slight cold-start delay, lowest cost |
| Batch processing | 300-600 seconds | Faster back-to-back jobs, higher idle cost |
| Heavy/continuous use | 600+ seconds | No cold-starts, highest cost |

### How to Change Idle Timeout

1. Go to RunPod dashboard
2. Navigate to your Serverless endpoint
3. Edit the endpoint settings
4. Adjust "Idle Timeout" to desired value (in seconds)
5. Save changes

### Cost Optimization Tips

- Set idle timeout to **30-60 seconds** for occasional transcription jobs
- Only increase timeout if processing multiple files in quick succession
- The TRANSCRIBE.IO app displays active GPU time only — it doesn't account for your serverless idle timeout
