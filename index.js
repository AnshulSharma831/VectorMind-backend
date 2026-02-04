//Phasse 1: Load Pdf, do chunking, do vector embedding, store the vectors in vector database,implement langchain

import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { Pinecone } from '@pinecone-database/pinecone';

import { PineconeStore } from '@langchain/pinecone';
import * as dotenv from 'dotenv';
dotenv.config();

async function indexPDF() {
  try {
    //reading pdf
    const PDF_PATH = './dsa.pdf'; // change to match your actual filename
    const pdfLoader = new PDFLoader(PDF_PATH);
    const rawDocs = await pdfLoader.load();
    console.log("pdf loaded...");

    //chunking for vector database
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });
    const chunkedDocs = await textSplitter.splitDocuments(rawDocs);
    console.log("chunking done...");
    
    //vector embedding

    const embeddings = new GoogleGenerativeAIEmbeddings({
      apiKey: process.env.GEMINI_API_KEY,
      model: 'text-embedding-004',
    });
    console.log("embedding model configured...");

    //database configuration
    //Initialize Pinecone Client
    if (!process.env.PINECONE_API_KEY) {
      throw new Error('PINECONE_API_KEY is not set in environment variables');
    }
    if (!process.env.PINECONE_INDEX_NAME) {
      throw new Error('PINECONE_INDEX_NAME is not set in environment variables');
    }
    const pinecone = new Pinecone({
      apiKey: process.env.PINECONE_API_KEY,
    });
    const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);
    console.log("pinecone configured...");

    //langchain - tell me chunking, embedding and vector database and i will do everything for you

    await PineconeStore.fromDocuments(chunkedDocs, embeddings, {
      pineconeIndex,
      maxConcurrency: 5, //select 5 out of 277 and store it in database
    });
    console.log("data stored successfully...");

  } catch (err) {
    console.error('Error while loading PDF:', err);
  }
}

indexPDF();