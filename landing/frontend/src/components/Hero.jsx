function Hero() {
  return (
    <header className="hero">
      <h1>LLM-Router</h1>
      <p className="tagline">One API for All LLM Providers</p>
      <div className="code-example">
        <pre>
          <code>
            <span className="keyword">from</span> gateway <span className="keyword">import</span> chat_complete{'\n'}
            <span className="keyword">from</span> models <span className="keyword">import</span> ChatCompletionRequest, Message{'\n'}
            {'\n'}
            request = ChatCompletionRequest({'\n'}
            {'    '}model=<span className="string">"gpt-4"</span>,{'\n'}
            {'    '}messages=[Message(role=<span className="string">"user"</span>, content=<span className="string">"Hello!"</span>)],{'\n'}
            ){'\n'}
            {'\n'}
            <span className="comment"># Switch providers with a single line change</span>{'\n'}
            response = chat_complete({'\n'}
            {'    '}provider=<span className="string">"openai"</span>,  <span className="comment"># or "anthropic", "google"</span>{'\n'}
            {'    '}api_key=<span className="string">"your-api-key"</span>,{'\n'}
            {'    '}request=request,{'\n'}
            )
          </code>
        </pre>
      </div>
    </header>
  )
}

export default Hero
