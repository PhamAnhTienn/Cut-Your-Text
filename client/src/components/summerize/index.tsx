import React, { useState } from 'react';
import './index.css';

const Summarize: React.FC = () => {
    const [text, setText] = useState<string>('');
    const [length, setLength] = useState<string>('Short');
    const [summary, setSummary] = useState<string>(''); 

    const handleSummarize = async () => {
        try {
            const response = await fetch( 'http://localhost:8080/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body : JSON.stringify({text}),
            });

            if ( response.ok ) {
                const data = await response.json();
                setSummary(data.summary);
            } else {
                console.error('Error:', response.statusText);
            }

        } catch (error) {
            console.error('Error', error)
        }

    };

    return (
        <div className="summarize-container">
            <div className="input-summary-wrapper">
                <div className="summarize-input">
                    <div className="summarize-length">
                        <label>Summary Length: </label>
                        <input
                            type="range"
                            min="0"
                            max="100"
                            value={length === 'Short' ? 0 : 100}
                            onChange={(e) =>
                                setLength(e.target.value === '0' ? 'Short' : 'Long')
                            }
                        />
                        <span>{length}</span>
                    </div>

                    <textarea
                        className="summarize-textarea"
                        placeholder="Enter or paste your text and press &quot;Summarize.&quot;"
                        value={text}
                        onChange={(e) => setText(e.target.value)}
                    />

                    <button
                        className="summarize-btn"
                        onClick={handleSummarize}
                    >
                        Summarize
                    </button>
                </div>

                <div className="summary-result">
                    <h3>Summary Result:</h3>
                    <p>{summary}</p>
                </div>
            </div>
        </div>
    );
};

export default Summarize;
