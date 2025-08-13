function cleanText(text) {
    if (!text) return "";
    text = text.toLowerCase();
    text = text.replace(/@[\w]+/g, '');
    text = text.replace(/https?:\/\/\S+|www\.\S+/g, '');
    text = text.replace(/<.*?>/g, '');
    text = text.replace(/[^a-zA-Z\s]/g, '');
    text = text.replace(/\s+/g, ' ').trim();
    return text;
}

function analyzeSentiment() {
    const inputText = document.getElementById('textInput').value;
    const resultContainer = document.getElementById('resultContainer');
    const sentimentResult = document.getElementById('sentimentResult');
    const confidenceFill = document.getElementById('confidenceFill');
    const confidenceText = document.getElementById('confidenceText');

    if (!inputText.trim()) {
        alert('Please enter some text to analyze!');
        return;
    }

    const cleanedText = cleanText(inputText);

    const positiveWords = ['love', 'best', 'amazing', 'great', 'excellent', 'perfect', 'incredible', 'awesome', 'good', 'nice', 'fantastic', 'wonderful', 'brilliant', 'outstanding'];
    const negativeWords = ['hate', 'terrible', 'awful', 'sucks', 'horrible', 'worst', 'bad', 'disgusting', 'broken', 'never', 'disappointed', 'angry', 'frustrated'];

    let positiveScore = 0;
    let negativeScore = 0;

    positiveWords.forEach(word => {
        if (cleanedText.includes(word)) positiveScore += Math.random() * 0.3 + 0.7;
    });

    negativeWords.forEach(word => {
        if (cleanedText.includes(word)) negativeScore += Math.random() * 0.3 + 0.7;
    });

    let sentiment, confidence, color;

    if (positiveScore > negativeScore && positiveScore > 0.3) {
        sentiment = 'Positive';
        confidence = Math.min(0.75 + Math.random() * 0.2, 0.95);
        color = '#4CAF50';
    } else if (negativeScore > positiveScore && negativeScore > 0.3) {
        sentiment = 'Negative';
        confidence = Math.min(0.70 + Math.random() * 0.25, 0.95);
        color = '#f44336';
    } else {
        sentiment = 'Neutral';
        confidence = 0.60 + Math.random() * 0.25;
        color = '#ff9800';
    }

    sentimentResult.innerHTML = `Predicted Sentiment: <span class="sentiment-${sentiment.toLowerCase()}">${sentiment}</span>`;
    confidenceFill.style.width = `${confidence * 100}%`;
    confidenceFill.style.backgroundColor = color;
    confidenceText.textContent = `Confidence: ${(confidence * 100).toFixed(1)}%`;

    resultContainer.classList.add('show');
}

document.getElementById('textInput').addEventListener('keypress', function(e) {
    if (e.key === 'Enter' && e.ctrlKey) {
        analyzeSentiment();
    }
});
