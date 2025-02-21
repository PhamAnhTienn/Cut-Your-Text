import React, { useState } from 'react';
import {
  Box,
  Button,
  CircularProgress,
  Container,
  Fade,
  Paper,
  TextField,
  Typography,
} from '@mui/material';

const Summarize: React.FC = () => {
  const [text, setText] = useState<string>('');
  const [summary, setSummary] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);

  const handleSummarize = async () => {
    if (!text.trim()) return;
    setIsLoading(true);
    try {
      const response = await fetch('http://localhost:8080/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      if (response.ok) {
        const data = await response.json();
        setSummary(data.summary);
      } else {
        console.error('Error:', response.statusText);
      }
    } catch (error) {
      console.error('Error', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Container maxWidth="md" sx={{ py: 4 }}>
      <Paper elevation={3} sx={{ p: 3, mb: 2, textAlign: 'center' }}>
        <Typography variant="h4" gutterBottom>
            Dialogue Summarization
        </Typography>
        <Typography variant="subtitle1" mb={2}>
          Paste your text below and let AI do the magic ✨
        </Typography>

        <TextField
          multiline
          rows={8}
          variant="outlined"
          placeholder="Enter or paste your text here..."
          fullWidth
          value={text}
          onChange={(e) => setText(e.target.value)}
          sx={{ mb: 2 }}
        />

        <Box position="relative" display="inline-block">
          <Button
            variant="contained"
            color="primary"
            disabled={isLoading || !text.trim()}
            onClick={handleSummarize}
          >
            {isLoading ? 'Loading...' : 'Summarize'}
          </Button>
          {isLoading && (
            <CircularProgress
              size={24}
              sx={{
                color: 'primary.main',
                position: 'absolute',
                top: '50%',
                left: '50%',
                marginTop: '-12px',
                marginLeft: '-12px',
              }}
            />
          )}
        </Box>
      </Paper>

      <Fade in={!!summary}>
        <Paper elevation={2} sx={{ p: 3, mt: 2 }}>
          <Typography variant="h5" gutterBottom>
            Summary Result
          </Typography>
          <Typography variant="body1">
            {summary || 'Your summary will appear here...'}
          </Typography>
        </Paper>
      </Fade>
    </Container>
  );
};

export default Summarize;
