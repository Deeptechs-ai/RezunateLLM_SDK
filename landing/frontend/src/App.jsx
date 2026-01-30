import Hero from './components/Hero'
import Features from './components/Features'
import Chat from './components/Chat'
import WaitlistForm from './components/WaitlistForm'
import Footer from './components/Footer'

function App() {
  return (
    <div className="container">
      <Hero />
      <Features />
      <Chat />
      {/* <WaitlistForm /> */}
      <Footer />
    </div>
  )
}

export default App
