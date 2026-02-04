function Footer() {
  const currentYear = new Date().getFullYear()

  return (
    <footer className="footer">
      <p>&copy; {currentYear} LLM-Router. All rights reserved.</p>
    </footer>
  )
}

export default Footer
