export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

export async function GET() {
  const pythonBackend = 'http://127.0.0.1:8000/logs';

  try {
    const response = await fetch(pythonBackend, {
      headers: {
        'Accept': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
      },
    });

    if (!response.ok) {
      return new Response('Failed to connect to log stream', { status: 500 });
    }

    // Create a TransformStream to pass through the SSE data
    const { readable, writable } = new TransformStream();

    // Pipe the response body to the writable stream
    response.body?.pipeTo(writable);

    // Return the readable stream with proper SSE headers
    return new Response(readable, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache, no-transform',
        'Connection': 'keep-alive',
        'X-Accel-Buffering': 'no', // Disable buffering in nginx if present
      },
    });
  } catch (error) {
    console.error('Error connecting to log stream:', error);
    return new Response('Failed to connect to log stream', { status: 500 });
  }
}
