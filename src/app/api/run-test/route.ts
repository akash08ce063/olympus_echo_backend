import { spawn } from 'child_process';
import fs from 'fs';
import { NextResponse } from 'next/server';
import os from 'os';
import path from 'path';

export async function POST(request: Request) {
  try {
    const body = await request.json()
    const { agentId, agentPhoneNumber, testerAgents, scenarios } = body

    if (!agentId || !agentPhoneNumber || !testerAgents || !scenarios) {
      return NextResponse.json(
        { error: 'Missing required fields: agentId, agentPhoneNumber, testerAgents, scenarios' },
        { status: 400 }
      )
    }

    // Agent validation (optional, can just trust the phone number if needed, or keeping basic checks)
    // For now we trust the phone number passed from the client which came from our /agents API


    // 2. Create Config File
    const overridePhoneNumber = process.env.TEST_PHONE_NUMBER
    const config = {
        phone_number_to_call: overridePhoneNumber || agentPhoneNumber,
        agents: testerAgents,
        scenarios: scenarios
    }

    const tempDir = os.tmpdir()
    const configPath = path.join(tempDir, `fixa_config_${crypto.randomUUID()}.json`)

    fs.writeFileSync(configPath, JSON.stringify(config, null, 2))

    // 3. Prepare Python Script Execution
    const scriptPath = path.resolve(process.cwd(), 'scripts/fixa_runner.py')

    const pythonArgs = [
      scriptPath,
      `--config_file=${configPath}`
    ]

    // 4. Spawn Child Process and wrap in Promise
    const runPythonScript = () => new Promise<{ output: unknown, error?: string }>((resolve, reject) => {

      // Use venv python if available, otherwise fallback to python3
      const venvPython = path.join(process.cwd(), '.venv/bin/python')
      const pythonCmd = fs.existsSync(venvPython) ? venvPython : 'python3'

      const pythonProcess = spawn(pythonCmd, pythonArgs)

      let stdoutData = ''
      let stderrData = ''

      pythonProcess.stdout.on('data', (data) => {
        stdoutData += data.toString()
      })

      pythonProcess.stderr.on('data', (data) => {
        stderrData += data.toString()
      })

      pythonProcess.on('close', (code) => {
        // Cleanup config file
        try {
            if (fs.existsSync(configPath)) {
                fs.unlinkSync(configPath)
            }
        } catch (e) {
            console.warn('Failed to delete temp config file:', e)
        }

        if (code !== 0) {
          console.error(`Python script exited with code ${code}`)
          console.error(`Stderr: ${stderrData}`)
          reject(`Script failed: ${stderrData}`)
          return
        }

        try {
          // Find the last line that looks like a JSON object
          const lines = stdoutData.trim().split('\n')
          let result = null
          for (let i = lines.length - 1; i >= 0; i--) {
            try {
               const potentialJson = JSON.parse(lines[i])
               if (potentialJson && (typeof potentialJson === 'object')) {
                 result = potentialJson
                 break
               }
            } catch {
              // Not JSON, continue searching
            }
          }

          if (result) {
            resolve({ output: result })
          } else {
             // If we couldn't find valid JSON, fail with the raw output
             console.error('Failed to find JSON in Python output:', stdoutData)
             reject(`Attributes parsing error: Could not parse JSON from output. Raw output: ${stdoutData}`)
          }

        } catch (e) {
          console.error('Failed to parse Python output:', stdoutData)
          reject(`Attributes parsing error: ${e}`)
        }
      })

      pythonProcess.on('error', (err) => {
        reject(`Process spawn error: ${err.message}`)
      })
    })

    const result = await runPythonScript()
    console.log('DEBUG: Python script output structure:', JSON.stringify(result.output, null, 2))
    console.log('DEBUG: Config content:', JSON.stringify(config, null, 2))

    const output = result.output as Record<string, unknown>

    if (output && output.error) {
      return NextResponse.json(output, { status: 500 })
    }

    return NextResponse.json(output)

  } catch (error: unknown) {
    console.error('Error running test:', error)
    const errorMessage = error instanceof Error ? error.message : 'Internal server error'
    return NextResponse.json(
      { error: errorMessage },
      { status: 500 }
    )
  }
}

