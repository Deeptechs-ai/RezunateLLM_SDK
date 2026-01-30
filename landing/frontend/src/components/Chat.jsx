import { useState, useRef, useEffect } from 'react'

function Chat() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')
  const [sessionId] = useState(() => crypto.randomUUID())
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return

    const userMessage = { role: 'user', content: input.trim() }
    const newMessages = [...messages, userMessage]
    setMessages(newMessages)
    setInput('')
    setIsLoading(true)
    setError('')

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: newMessages,
          session_id: sessionId,
        }),
      })

      const result = await response.json()

      if (!response.ok) {
        setError(result.detail || 'Something went wrong')
        return
      }

      setMessages([...newMessages, { role: 'assistant', content: result.content }])
    } catch (err) {
      setError('Network error. Please try again.')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <section className="chat-section">
      <h2>Try It Now</h2>
      <p>Experience LLM-Router with a live demo powered by OpenAI.</p>

      <div className="chat-container">
        <div className="chat-messages">
          {messages.length === 0 && (
            <div className="chat-empty">
              Send a message to start chatting
            </div>
          )}
          {messages.map((msg, index) => (
            <div key={index} className={`chat-message ${msg.role}`}>
              <div className="message-role">{msg.role === 'user' ? 'You' : 'Assistant'}</div>
              <div className="message-content">{msg.content}</div>
            </div>
          ))}
          {isLoading && (
            <div className="chat-message assistant">
              <div className="message-role">Assistant</div>
              <div className="message-content typing">Thinking...</div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {error && <div className="chat-error">{error}</div>}

        <form className="chat-input-form" onSubmit={handleSubmit}>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type a message..."
            disabled={isLoading}
            className="chat-input"
          />
          <button type="submit" disabled={isLoading || !input.trim()} className="chat-send">
            {isLoading ? '...' : 'Send'}
          </button>
        </form>
      </div>
    </section>
  )
}

export default Chat
