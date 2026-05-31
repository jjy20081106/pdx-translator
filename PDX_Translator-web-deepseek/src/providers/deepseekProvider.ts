import { buildPrompt } from '../ollama/buildPrompt'
import type { TranslationProvider } from './types'

type DeepSeekMessage = {
  role: 'user' | 'assistant' | 'system'
  content: string
}

type DeepSeekResponse = {
  choices?: Array<{
    message?: {
      content?: string
    }
    finish_reason?: string
  }>
  error?: {
    message?: string
  }
}

function extractDeepSeekText(data: DeepSeekResponse) {
  const text = data.choices?.[0]?.message?.content?.trim()

  if (!text) {
    throw new Error(data.error?.message ?? 'DeepSeek response did not include translated text.')
  }

  return text
}

async function readDeepSeekError(response: Response) {
  try {
    const data = (await response.json()) as DeepSeekResponse
    return data.error?.message ?? `DeepSeek returned HTTP ${response.status}.`
  } catch {
    return `DeepSeek returned HTTP ${response.status}.`
  }
}

async function createDeepSeekResponse(
  prompt: string,
  settings: Parameters<TranslationProvider['translateBatch']>[1],
  signal?: AbortSignal,
) {
  const messages: DeepSeekMessage[] = [{ role: 'user', content: prompt }]

  const response = await fetch('https://api.deepseek.com/chat/completions', {
    method: 'POST',
    signal,
    headers: {
      Authorization: `Bearer ${settings.apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: settings.model,
      messages,
      temperature: settings.temperature,
      top_p: settings.topP,
      max_tokens: 8192,
      stream: false,
    }),
  })

  if (!response.ok) {
    throw new Error(await readDeepSeekError(response))
  }

  return extractDeepSeekText((await response.json()) as DeepSeekResponse)
}

export const deepseekProvider: TranslationProvider = {
  id: 'deepseek',
  label: 'DeepSeek API',
  defaultModel: 'deepseek-chat',
  requiresApiKey: true,
  async checkConnection(settings) {
    if (!settings.apiKey.trim()) {
      return { ok: false, label: 'DeepSeek API', error: 'API key is required.' }
    }

    try {
      await createDeepSeekResponse('Return OK only.', {
        ...settings,
        temperature: 0,
      })

      return { ok: true, label: 'DeepSeek API', detail: 'Test request completed.' }
    } catch (error) {
      return {
        ok: false,
        label: 'DeepSeek API',
        error: error instanceof Error ? error.message : 'Unable to call DeepSeek API.',
      }
    }
  },
  translateBatch(batch, settings, signal, retryInstructions) {
    return createDeepSeekResponse(
      buildPrompt(batch, {
        sourceLanguage: settings.sourceLanguage,
        targetLanguage: settings.targetLanguage,
        customInstructions: settings.customInstructions,
        glossaryEntries: settings.glossaryEntries,
        retryInstructions,
      }),
      settings,
      signal,
    )
  },
}
