import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Summarize from './components/summerize';

const App: React.FC = () => {
    return (
        <Router>
          <Routes>
            <Route path="/summarize" element={<Summarize />} />
          </Routes>
        </Router>
    );
};

export default App;
