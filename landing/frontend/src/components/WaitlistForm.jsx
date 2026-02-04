import { useState } from 'react'

function WaitlistForm() {
  const [formData, setFormData] = useState({
    email: '',
    name: '',
    company: '',
    use_case: '',
  })
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [message, setMessage] = useState({ type: '', text: '' })

  const handleChange = (e) => {
    const { name, value } = e.target
    setFormData((prev) => ({ ...prev, [name]: value }))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setIsSubmitting(true)
    setMessage({ type: '', text: '' })

    const data = {
      email: formData.email,
      name: formData.name || null,
      company: formData.company || null,
      use_case: formData.use_case || null,
    }

    try {
      const response = await fetch('/api/waitlist', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      })

      const result = await response.json()

      if (response.ok) {
        setMessage({ type: 'success', text: "You're on the list! We'll be in touch soon." })
        setFormData({ email: '', name: '', company: '', use_case: '' })
      } else if (response.status === 409) {
        setMessage({ type: 'error', text: result.detail || 'This email is already registered.' })
      } else if (response.status === 422) {
        const errorDetail = result.detail?.[0]?.msg || 'Please check your input and try again.'
        setMessage({ type: 'error', text: errorDetail })
      } else {
        setMessage({ type: 'error', text: 'Something went wrong. Please try again later.' })
      }
    } catch (error) {
      console.error('Submission error:', error)
      setMessage({ type: 'error', text: 'Network error. Please check your connection and try again.' })
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <section className="waitlist">
      <h2>Join the Waitlist</h2>
      <p>Be the first to know when we launch. Get early access and exclusive updates.</p>

      <form className="waitlist-form" onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="email">Email <span className="required">*</span></label>
          <input
            type="email"
            id="email"
            name="email"
            required
            placeholder="you@example.com"
            value={formData.email}
            onChange={handleChange}
          />
        </div>

        <div className="form-group">
          <label htmlFor="name">Name</label>
          <input
            type="text"
            id="name"
            name="name"
            placeholder="Your name"
            value={formData.name}
            onChange={handleChange}
          />
        </div>

        <div className="form-group">
          <label htmlFor="company">Company</label>
          <input
            type="text"
            id="company"
            name="company"
            placeholder="Your company"
            value={formData.company}
            onChange={handleChange}
          />
        </div>

        <div className="form-group">
          <label htmlFor="use_case">How do you plan to use LLM-Router?</label>
          <textarea
            id="use_case"
            name="use_case"
            rows="3"
            placeholder="Tell us about your use case..."
            value={formData.use_case}
            onChange={handleChange}
          />
        </div>

        <button type="submit" className="submit-btn" disabled={isSubmitting}>
          {isSubmitting ? 'Joining...' : 'Join Waitlist'}
        </button>

        {message.text && (
          <div className={`form-message ${message.type}`}>
            {message.text}
          </div>
        )}
      </form>
    </section>
  )
}

export default WaitlistForm
