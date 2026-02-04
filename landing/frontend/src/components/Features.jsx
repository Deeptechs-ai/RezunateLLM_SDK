function Features() {
  const features = [
    {
      icon: '\u{1F504}',
      title: 'Unified Format',
      description: 'Consistent request and response format across all providers. No more provider-specific code.',
    },
    {
      icon: '\u{26A1}',
      title: 'Easy Switching',
      description: 'Switch between providers with a single line change. No refactoring required.',
    },
  ]

  return (
    <section className="features">
      {features.map((feature, index) => (
        <div key={index} className="feature">
          <div className="feature-icon">{feature.icon}</div>
          <h3>{feature.title}</h3>
          <p>{feature.description}</p>
        </div>
      ))}
    </section>
  )
}

export default Features
