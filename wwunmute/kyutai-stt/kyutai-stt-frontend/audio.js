
let recorder = null;
let socket = null;
let warmupComplete = false;
let completedSentences = [];
let pendingSentence = '';
let vadMessageCount = 0;
let textMessageCount = 0;

const getBaseURL = () => {
    const currentURL = new URL(window.location.href);
    let hostname = currentURL.hostname;
    hostname = hostname.replace('-ui', '-stt-api');
    const wsProtocol = currentURL.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${wsProtocol}//${hostname}/ws`;
}

const updateTextOutput = () => {
    const container = document.getElementById('text-output');
    if (!container) return;

    const allSentences = [...completedSentences];
    if (pendingSentence) {
        allSentences.push(pendingSentence);
    }

    let content = '';

    if (warmupComplete) {
        content += allSentences.map(sentence =>
            `<p class="text-gray-300 my-2">${sentence}</p>`
        ).reverse().join('');
    } else {
        content = '<p class="text-gray-400 animate-pulse">Warming up model...</p>';
    }

    container.innerHTML = content;
    container.scrollTop = container.scrollHeight;
};

const startRecording = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    recorder = new Recorder({
        encoderPath: "https://cdn.jsdelivr.net/npm/opus-recorder@latest/dist/encoderWorker.min.js",
        streamPages: true,
        encoderApplication: 2049,
        encoderFrameSize: 80,
        encoderSampleRate: 24000,
        maxFramesPerPage: 1,
        numberOfChannels: 1,
    });

    recorder.ondataavailable = async (arrayBuffer) => {
        if (socket && socket.readyState === WebSocket.OPEN) {
            await socket.send(arrayBuffer);
        }
    };

    recorder.start().then(() => {
        console.log("Recording started");
        recorder.setRecordingGain(1);
    });

    const analyzerContext = new (window.AudioContext || window.webkitAudioContext)();
    const analyzer = analyzerContext.createAnalyser();
    analyzer.fftSize = 256;
    const sourceNode = analyzerContext.createMediaStreamSource(stream);
    sourceNode.connect(analyzer);

    const processAudio = () => {
        const dataArray = new Uint8Array(analyzer.frequencyBinCount);
        analyzer.getByteFrequencyData(dataArray);
        requestAnimationFrame(processAudio);
    };
    processAudio();
};

const initApp = () => {
    const endpoint = getBaseURL();
    console.log("Connecting to", endpoint);
    socket = new WebSocket(endpoint);

    socket.onopen = () => {
        console.log("WebSocket connection opened");
        startRecording();
        warmupComplete = true;
        updateTextOutput();
    };

    socket.onmessage = async (event) => {
        // data is a blob, convert to array buffer
        const arrayBuffer = await event.data.arrayBuffer();
        const view = new Uint8Array(arrayBuffer);
        const tag = view[0];
        const payload = arrayBuffer.slice(1);

        // Pretty logging for transcription data
        const transcriptionData = {
            timestamp: new Date().toISOString(),
            rawData: {
                arrayBufferSize: arrayBuffer.byteLength,
                tag: tag,
                tagHex: `0x${tag.toString(16).padStart(2, '0')}`,
                payloadSize: payload.byteLength,
                payloadBytes: Array.from(new Uint8Array(payload)).map(b => `0x${b.toString(16).padStart(2, '0')}`).join(' ')
            }
        };

        if (tag === 1) {
            // text data
            textMessageCount++;
            const decoder = new TextDecoder();
            const text = decoder.decode(payload);

            // Add text information to the data structure
            transcriptionData.textData = {
                decodedText: text,
                textLength: text.length,
                isWhitespace: text.trim() === '',
                isPunctuation: /[.!?]/.test(text)
            };

            // Add sentence state information
            transcriptionData.sentenceState = {
                beforeUpdate: {
                    pendingSentence: pendingSentence,
                    pendingLength: pendingSentence.length,
                    completedCount: completedSentences.length
                }
            };

            pendingSentence += text;

            transcriptionData.sentenceState.afterUpdate = {
                pendingSentence: pendingSentence,
                pendingLength: pendingSentence.length,
                willComplete: pendingSentence.endsWith('.') || pendingSentence.endsWith('!') || pendingSentence.endsWith('?')
            };
        } else if (tag === 2) {
            // VAD data
            vadMessageCount++;
            const decoder = new TextDecoder();
            const vadJson = decoder.decode(payload);
            try {
                const vadData = JSON.parse(vadJson);
                transcriptionData.vadData = {
                    probability: vadData.probability,
                    hasVadHeads: vadData.has_vad_heads,
                    isActive: vadData.probability !== null ? vadData.probability > 0.5 : null,
                    confidenceLevel: vadData.probability !== null ?
                        (vadData.probability > 0.8 ? 'high' : vadData.probability > 0.5 ? 'medium' : 'low') :
                        'unknown',
                    rawJson: vadJson
                };
            } catch (e) {
                transcriptionData.vadData = {
                    error: `Failed to parse VAD JSON: ${e.message}`,
                    rawJson: vadJson
                };
            }
        } else {
            transcriptionData.unknownTag = {
                message: `Received unknown tag: ${tag}`,
                payloadPreview: Array.from(new Uint8Array(payload.slice(0, 10))).map(b => `0x${b.toString(16).padStart(2, '0')}`).join(' ')
            };
        }

        // Pretty console logging with styling
        console.group(`🎤 Transcription Data - ${transcriptionData.timestamp}`);
        console.log('📊 Raw Data:', transcriptionData.rawData);
        console.log(`📈 Message Counts: Text=${textMessageCount}, VAD=${vadMessageCount}`);

        if (transcriptionData.textData) {
            console.log('📝 Text Data:', transcriptionData.textData);
            console.log('📋 Sentence State:', transcriptionData.sentenceState);

            // Color-coded text display
            const textDisplay = transcriptionData.textData.decodedText === ' ' ?
                `"${transcriptionData.textData.decodedText}" (space)` :
                `"${transcriptionData.textData.decodedText}"`;

            if (transcriptionData.textData.isPunctuation) {
                console.log(`🔴 New Token: ${textDisplay} (punctuation - sentence may complete)`);
            } else if (transcriptionData.textData.isWhitespace) {
                console.log(`⚪ New Token: ${textDisplay} (whitespace)`);
            } else {
                console.log(`🟢 New Token: ${textDisplay} (word fragment)`);
            }
        }

        if (transcriptionData.vadData) {
            console.log('🎯 VAD Data:', transcriptionData.vadData);

            if (transcriptionData.vadData.error) {
                console.error(`❌ VAD Error: ${transcriptionData.vadData.error}`);
            } else if (!transcriptionData.vadData.hasVadHeads) {
                console.warn(`⚠️ VAD: No VAD heads available from model`);
            } else if (transcriptionData.vadData.probability === null) {
                console.warn(`⚠️ VAD: Null probability received`);
            } else {
                // Color-coded VAD display
                const vadProbability = transcriptionData.vadData.probability;
                const vadDisplay = `${(vadProbability * 100).toFixed(1)}%`;

                if (transcriptionData.vadData.confidenceLevel === 'high') {
                    console.log(`🔊 VAD: ${vadDisplay} (HIGH confidence - voice very active)`);
                } else if (transcriptionData.vadData.confidenceLevel === 'medium') {
                    console.log(`🔉 VAD: ${vadDisplay} (MEDIUM confidence - voice detected)`);
                } else {
                    console.log(`🔇 VAD: ${vadDisplay} (LOW confidence - likely silence)`);
                }

                if (transcriptionData.vadData.isActive) {
                    console.log(`✅ Voice Activity: ACTIVE (probability > 50%)`);
                } else {
                    console.log(`🚫 Voice Activity: INACTIVE (probability ≤ 50%)`);
                }
            }
        }

        if (transcriptionData.unknownTag) {
            console.warn('⚠️ Unknown Tag:', transcriptionData.unknownTag);
        }

        console.groupEnd();

        if (tag === 1) {
            // text data processing (existing logic)
            if (pendingSentence.endsWith('.') || pendingSentence.endsWith('!') || pendingSentence.endsWith('?')) {
                completedSentences.push(pendingSentence);
                
                // Log completed sentence
                console.log(`✅ Sentence Completed: "${pendingSentence}"`);
                
                pendingSentence = '';
            }
        }

        updateTextOutput();
    };

    socket.onclose = () => {
        console.log("WebSocket connection closed");
    };

    updateTextOutput();
};

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initApp);
} else {
    initApp();
}