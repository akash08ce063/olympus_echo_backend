import { NextResponse } from 'next/server';

export async function GET() {
  try {
    const backendUrl = process.env.BACKEND_URL
    const authToken = process.env.AUTH_TOKEN

    if (!backendUrl) {
      throw new Error('BACKEND_URL is not defined')
    }

    if (!authToken) {
      throw new Error('AUTH_TOKEN is not defined')
    }

    const response = await fetch(`${backendUrl}/api/v1/phone/mapping`, {
      headers: {
        'Authorization': `Bearer ${authToken}`,
        'Content-Type': 'application/json',
      },
    })

    if (!response.ok) {
      throw new Error(`Backend responded with status: ${response.status}`)
    }

    const data = await response.json()

    // Transform the response to extract agent information
    if (!data.success || !data.mappings) {
      throw new Error('Invalid response format from backend')
    }

    const agents: { id: string; name: string; phoneNumber: string }[] = []
    const seenIds = new Set<string>()

    for (const mapping of data.mappings) {
      if (!seenIds.has(mapping.agent_id)) {
        seenIds.add(mapping.agent_id)
        agents.push({
          id: mapping.agent_id,
          name: mapping.agent_name,
          phoneNumber: mapping.phone_number,
        })
      }
    }

    return NextResponse.json(agents)
  } catch (error: unknown) {
    console.error('Error fetching agents:', error)
    const errorMessage = error instanceof Error ? error.message : 'Internal server error'
    return NextResponse.json(
      { error: errorMessage },
      { status: 500 }
    )
  }
}
