import express from 'express';
import cors from 'cors';
import multer from 'multer';
import path from 'path';
import { fileURLToPath } from 'url';
import { promises as fs } from 'fs';
import * as dotenv from 'dotenv';
dotenv.config();

import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { Pinecone } from '@pinecone-database/pinecone';
import { PineconeStore } from '@langchain/pinecone';
import { GoogleGenAI } from "@google/genai";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 5001;

// Middleware
const allowedOrigins = [
  'http://localhost:3000',
  'http://localhost:5173',
  'https://vector-mind-frontend.vercel.app/'
];

app.use(cors({
  origin: function (origin, callback) {
    // Allow requests with no origin (mobile apps, curl, Postman)
    if (!origin) return callback(null, true);

    if (allowedOrigins.includes(origin)) {
      callback(null, true);
    } else {
      callback(new Error(`CORS blocked: ${origin}`));
    }
  },
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization'],
  credentials: true,
  optionsSuccessStatus: 200
}));

// IMPORTANT for preflight
app.options('*', cors());
app.use(express.json());
app.use(express.static('public'));

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, 'uploads/');
  },
  filename: (req, file, cb) => {
    cb(null, `pdf-${Date.now()}${path.extname(file.originalname)}`);
  }
});

const upload = multer({
  storage: storage,
  fileFilter: (req, file, cb) => {
    if (file.mimetype === 'application/pdf') {
      cb(null, true);
    } else {
      cb(new Error('Only PDF files are allowed!'), false);
    }
  },
  limits: { fileSize: 10 * 1024 * 1024 } // 10MB limit
});

// Store conversation history per session (in production, use Redis or database)
const conversationHistories = new Map();

// Initialize AI
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

// Helper function to get or create history for a session
function getHistory(sessionId) {
  if (!conversationHistories.has(sessionId)) {
    conversationHistories.set(sessionId, []);
  }
  return conversationHistories.get(sessionId);
}

// Helper function to clear history for a session
function clearHistory(sessionId) {
  conversationHistories.set(sessionId, []);
}

// Transform query with history
async function transformQuery(question, history) {
  try {
    const tempHistory = [...history];
    tempHistory.push({
      role: 'user',
      parts: [{ text: question }]
    });

    const response = await ai.models.generateContent({
      model: "gemini-3-flash-preview",
      contents: tempHistory,
      config: {
        systemInstruction: `You are a query rewriting expert. Based on the provided chat history, rephrase the "Follow Up user Question" into a complete, standalone question that can be understood without the chat history.
        Only output the rewritten question and nothing else.`,
      },
    });

    return response.text;
  } catch (error) {
    console.error('Query transformation error:', error);
    // If transformation fails, return original question
    return question;
  }
}

// Helper function to delete all vectors from Pinecone index
async function deleteAllVectors(pineconeIndex) {
  try {
    console.log("🗑️  Attempting to delete all existing vectors from Pinecone...");
    
    // Get index stats first
    const stats = await pineconeIndex.describeIndexStats();
    const totalVectors = stats.totalRecordCount || 0;
    
    if (totalVectors === 0) {
      console.log("✅ No existing vectors to delete");
      return;
    }
    
    console.log(`Found ${totalVectors} vectors in index`);
    
    // Use deleteAll if available (newer Pinecone versions)
    try {
      // For serverless indexes, delete all vectors in default namespace
      await pineconeIndex.deleteAll();
      console.log("✅ All vectors deleted using deleteAll()");
    } catch (deleteAllError) {
      console.log("⚠️  deleteAll() not available, using alternative method...");
      
      // Alternative: Delete by fetching IDs
      try {
        // List all IDs (up to 10000)
        const listResponse = await pineconeIndex.listPaginated({ limit: 10000 });
        
        if (listResponse && listResponse.vectors && listResponse.vectors.length > 0) {
          const ids = listResponse.vectors.map(v => v.id);
          await pineconeIndex.deleteMany(ids);
          console.log(`✅ Deleted ${ids.length} vectors`);
        } else {
          console.log("✅ No vectors to delete");
        }
      } catch (listError) {
        console.warn("⚠️  Could not list/delete vectors:", listError.message);
        console.log("📝 Proceeding anyway - new vectors will be added");
      }
    }
    
    // Wait for deletion to propagate
    await new Promise(resolve => setTimeout(resolve, 1000));
  } catch (error) {
    console.warn("⚠️  Warning during vector deletion:", error.message);
    console.log("📝 Proceeding with indexing anyway");
    // Don't throw - allow indexing to continue
  }
}

// Helper function to process documents in smaller batches
async function processBatchedDocuments(chunkedDocs, embeddings, pineconeIndex, batchSize = 50) {
  console.log(`💾 Processing ${chunkedDocs.length} chunks in batches of ${batchSize}...`);
  
  const totalBatches = Math.ceil(chunkedDocs.length / batchSize);
  
  for (let i = 0; i < chunkedDocs.length; i += batchSize) {
    const currentBatch = Math.floor(i / batchSize) + 1;
    const batch = chunkedDocs.slice(i, i + batchSize);
    
    console.log(`📦 Processing batch ${currentBatch}/${totalBatches} (${batch.length} chunks)...`);
    
    try {
      await PineconeStore.fromDocuments(batch, embeddings, {
        pineconeIndex,
        maxConcurrency: 3, // Reduced from 5 to avoid rate limits
      });
      
      console.log(`✅ Batch ${currentBatch}/${totalBatches} completed`);
      
      // Small delay between batches to avoid rate limits
      if (i + batchSize < chunkedDocs.length) {
        await new Promise(resolve => setTimeout(resolve, 500));
      }
    } catch (batchError) {
      console.error(`❌ Error in batch ${currentBatch}:`, batchError.message);
      throw batchError;
    }
  }
  
  console.log("✅ All batches processed successfully");
}

// Upload PDF endpoint
app.post('/api/upload-pdf', (req, res) => {
  upload.single('pdf')(req, res, (err) => {
    try {
      if (err) {
        // Multer errors (file too large, invalid type, etc.)
        const message =
          err.code === 'LIMIT_FILE_SIZE'
            ? 'File size must be less than 10MB'
            : err.message || 'File upload failed';
        return res.status(400).json({ error: message });
      }

      if (!req.file) {
        // Most common cause: request not sent as multipart/form-data or wrong field name
        return res.status(400).json({
          error: 'No PDF file uploaded (missing form field "pdf")',
        });
      }

      console.log(`✅ PDF uploaded: ${req.file.filename}`);
      return res.json({
        message: 'PDF uploaded successfully',
        filename: req.file.filename,
      });
    } catch (error) {
      console.error('Upload error:', error);
      return res.status(500).json({
        error: 'Failed to upload PDF',
        details: error.message,
      });
    }
  });
});

// Index PDF to Pinecone endpoint
app.post('/api/index-pdf', async (req, res) => {
  let currentStep = 'initialization';
  
  try {
    const { filename } = req.body;
    
    console.log('\n========================================');
    console.log('📥 Received index-pdf request');
    console.log('Filename:', filename);
    console.log('========================================\n');
    
    if (!filename) {
      return res.status(400).json({ error: 'Filename is required' });
    }

    const filePath = path.join(__dirname, 'uploads', filename);
    console.log('File path:', filePath);
    
    // Check if file exists
    currentStep = 'file_check';
    try {
      await fs.access(filePath);
      console.log('✅ File exists');
    } catch (err) {
      console.error('❌ File not found:', err);
      return res.status(404).json({ 
        error: 'PDF file not found', 
        details: `File path: ${filePath}`,
        step: currentStep
      });
    }

    console.log(`\n📚 Starting PDF indexing for: ${filename}`);
    
    // Load PDF
    currentStep = 'loading_pdf';
    console.log("📖 Loading PDF...");
    const pdfLoader = new PDFLoader(filePath);
    const rawDocs = await pdfLoader.load();
    console.log(`✅ PDF loaded: ${rawDocs.length} pages`);

    if (!rawDocs || rawDocs.length === 0) {
      return res.status(400).json({ 
        error: 'PDF is empty or could not be parsed',
        step: currentStep
      });
    }

    // Chunking
    currentStep = 'chunking';
    console.log("✂️  Chunking documents...");
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });
    const chunkedDocs = await textSplitter.splitDocuments(rawDocs);
    console.log(`✅ Chunking done: ${chunkedDocs.length} chunks`);

    if (!chunkedDocs || chunkedDocs.length === 0) {
      return res.status(400).json({ 
        error: 'No text chunks could be created from PDF',
        step: currentStep
      });
    }

    // Embeddings
    currentStep = 'configuring_embeddings';
    if (!process.env.GEMINI_API_KEY) {
      return res.status(500).json({ 
        error: 'GEMINI_API_KEY is not set in environment variables',
        step: currentStep
      });
    }

    console.log("🔢 Configuring embedding model...");
    const embeddings = new GoogleGenerativeAIEmbeddings({
      apiKey: process.env.GEMINI_API_KEY,
      model: 'text-embedding-004',
    });
    console.log("✅ Embedding model configured");

    // Pinecone setup
    currentStep = 'configuring_pinecone';
    if (!process.env.PINECONE_API_KEY) {
      return res.status(500).json({ 
        error: 'PINECONE_API_KEY is not set in environment variables',
        step: currentStep
      });
    }
    
    if (!process.env.PINECONE_INDEX_NAME) {
      return res.status(500).json({ 
        error: 'PINECONE_INDEX_NAME is not set in environment variables',
        step: currentStep
      });
    }

    console.log("🔧 Configuring Pinecone...");
    const pinecone = new Pinecone({
      apiKey: process.env.PINECONE_API_KEY,
    });
    const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);
    console.log("✅ Pinecone configured");

    // Delete all existing vectors before adding new ones
    currentStep = 'deleting_vectors';
    await deleteAllVectors(pineconeIndex);

    // Store new vectors using batched processing
    currentStep = 'storing_vectors';
    await processBatchedDocuments(chunkedDocs, embeddings, pineconeIndex, 50);

    // Clean up old PDF files (optional - keep only the latest)
    currentStep = 'cleanup';
    try {
      const files = await fs.readdir(path.join(__dirname, 'uploads'));
      const pdfFiles = files.filter(f => f.startsWith('pdf-') && f !== filename);
      for (const file of pdfFiles) {
        await fs.unlink(path.join(__dirname, 'uploads', file));
        console.log(`🗑️  Deleted old file: ${file}`);
      }
    } catch (cleanupError) {
      console.warn('⚠️  Warning: Could not clean up old files:', cleanupError.message);
    }

    console.log('\n✅ SUCCESS: PDF indexed successfully!\n');
    res.json({ 
      message: 'PDF indexed successfully and old vectors removed',
      chunks: chunkedDocs.length 
    });
  } catch (error) {
    console.error('\n❌ ========================================');
    console.error('ERROR OCCURRED DURING INDEXING');
    console.error('========================================');
    console.error(`Step: ${currentStep}`);
    console.error('Error name:', error.name);
    console.error('Error message:', error.message);
    console.error('Error stack:', error.stack);
    console.error('========================================\n');
    
    // Provide more specific error messages
    let errorMessage = error.message;
    let errorDetails = `Failed at step: ${currentStep}`;
    
    if (error.message.includes('ENOENT')) {
      errorMessage = 'PDF file not found. Please ensure the file was uploaded correctly.';
    } else if (error.message.includes('API key') || error.message.includes('authentication')) {
      errorMessage = 'API key error. Please check your GEMINI_API_KEY or PINECONE_API_KEY in .env file.';
    } else if (error.message.includes('index') || error.message.includes('Index')) {
      errorMessage = 'Pinecone index error. Please verify your PINECONE_INDEX_NAME exists.';
    } else if (error.message.includes('embedding') || error.message.includes('embed')) {
      errorMessage = 'Embedding generation failed. Please check your GEMINI_API_KEY.';
    } else if (error.message.includes('quota') || error.message.includes('rate limit')) {
      errorMessage = 'API rate limit exceeded. Please wait a moment and try again.';
    } else if (error.message.includes('network') || error.message.includes('ECONNREFUSED')) {
      errorMessage = 'Network error. Please check your internet connection.';
    }
    
    // Make sure we send a JSON response
    try {
      res.status(500).json({ 
        error: 'Failed to index PDF', 
        details: errorDetails,
        message: errorMessage,
        step: currentStep,
        originalError: error.message
      });
    } catch (sendError) {
      console.error('Failed to send error response:', sendError);
    }
  }
});

// Chat endpoint
app.post('/api/chat', async (req, res) => {
  try {
    const { question, sessionId } = req.body;

    if (!question) {
      return res.status(400).json({ error: 'Question is required' });
    }

    // Check environment variables
    if (!process.env.GEMINI_API_KEY) {
      return res.status(500).json({ 
        error: 'GEMINI_API_KEY is not set in environment variables' 
      });
    }
    
    if (!process.env.PINECONE_API_KEY) {
      return res.status(500).json({ 
        error: 'PINECONE_API_KEY is not set in environment variables' 
      });
    }
    
    if (!process.env.PINECONE_INDEX_NAME) {
      return res.status(500).json({ 
        error: 'PINECONE_INDEX_NAME is not set in environment variables' 
      });
    }

    // Generate a unique session ID if not provided
    const currentSessionId = sessionId || `session-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`;
    const history = getHistory(currentSessionId);

    // Transform query
    const queries = await transformQuery(question, history);

    // Validate transformed query
    if (!queries || typeof queries !== 'string' || queries.trim().length === 0) {
      return res.status(400).json({ 
        error: 'Query transformation failed or returned invalid result' 
      });
    }

    // Create embeddings
    const embeddings = new GoogleGenerativeAIEmbeddings({
      apiKey: process.env.GEMINI_API_KEY,
      model: 'text-embedding-004',
    });

    const queryVector = await embeddings.embedQuery(queries);

    // Search Pinecone
    const pinecone = new Pinecone({
      apiKey: process.env.PINECONE_API_KEY,
    });
    const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);
    const searchResults = await pineconeIndex.query({
      topK: 10,
      vector: queryVector,
      includeMetadata: true,
    });

    // Extract context
    const context = searchResults.matches
      .map(match => match.metadata?.text)
      .filter(text => text)
      .join("\n\n---\n\n");

    // Check if we have relevant context
    if (!context || context.trim().length === 0) {
      return res.json({
        response: "I couldn't find any relevant information in the document to answer your question. Please make sure a PDF has been uploaded and indexed.",
        sessionId: currentSessionId
      });
    }

    // Update history with rewritten query
    history.push({
      role: 'user',
      parts: [{ text: queries }]
    });

    // Generate response
    const response = await ai.models.generateContent({
      model: "gemini-3-flash-preview",
      contents: history,
      config: {
        systemInstruction: `You are a Retrieval-Augmented Generation Model, Your name is Enigma.
                You will be given a context of relevant information and a user question.
                Your task is to answer the user's question based ONLY on the provided context.
                If the answer is not in the context, you must say "I could not find the answer in the provided document."
                Keep your answers clear, concise, and educational.
                  
                Context: ${context}`,
      },
    });

    const responseText = response.text || "No response generated";

    // Update history with response
    history.push({
      role: 'model',
      parts: [{ text: responseText }]
    });

    res.json({ 
      response: responseText,
      sessionId: currentSessionId
    });
  } catch (error) {
    console.error('Chat error:', error);
    res.status(500).json({ error: 'Failed to process chat', details: error.message });
  }
});

// Clear chat history endpoint
app.post('/api/clear-history', (req, res) => {
  try {
    const { sessionId } = req.body;
    if (sessionId) {
      clearHistory(sessionId);
      res.json({ message: 'History cleared for session', sessionId });
    } else {
      res.status(400).json({ error: 'Session ID is required' });
    }
  } catch (error) {
    console.error('Clear history error:', error);
    res.status(500).json({ error: 'Failed to clear history', details: error.message });
  }
});

// Health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'OK', timestamp: new Date().toISOString() });
});

// Start server with error handling
const server = app.listen(PORT, async () => {
  console.log(`\n🚀 Server running on port ${PORT}`);
  console.log(`📍 Health check: http://localhost:${PORT}/api/health\n`);
  
  // Create uploads directory if it doesn't exist
  const uploadsDir = path.join(__dirname, 'uploads');
  try {
    await fs.access(uploadsDir);
  } catch {
    await fs.mkdir(uploadsDir, { recursive: true });
    console.log('📁 Created uploads directory');
  }
});

// Handle port already in use error
server.on('error', (error) => {
  if (error.code === 'EADDRINUSE') {
    console.error(`\n❌ Port ${PORT} is already in use!`);
    console.error(`\n💡 Solutions:`);
    console.error(`   1. Kill the process using port ${PORT}:`);
    console.error(`      netstat -ano | findstr :${PORT}`);
    console.error(`      taskkill /PID <PID> /F`);
    console.error(`\n   2. Change PORT in .env file to a different port (e.g., 5002)\n`);
    process.exit(1);
  } else {
    console.error('Server error:', error);
    process.exit(1);
  }
});

// Handle unhandled promise rejections
process.on('unhandledRejection', (reason, promise) => {
  console.error('❌ Unhandled Rejection at:', promise);
  console.error('Reason:', reason);
  // Don't exit - just log it
});

// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
  console.error('❌ Uncaught Exception:', error);
  console.error('Stack:', error.stack);
  // Don't exit - just log it
});