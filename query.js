import * as dotenv from 'dotenv';
dotenv.config();
import readlineSync from 'readline-sync';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { Pinecone } from '@pinecone-database/pinecone';
import { GoogleGenAI } from "@google/genai";

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
const History = []

async function transformQuery(question){
    try {
        History.push({
            role:'user',
            parts:[{text:question}]
        });  
        
        const response = await ai.models.generateContent({
            model: "gemini-2.5-flash",
            contents: History,
            config: {
              systemInstruction: `You are a query rewriting expert. Based on the provided chat history, rephrase the "Follow Up user Question" into a complete, standalone question that can be understood without the chat history.
            Only output the rewritten question and nothing else.
              `,
            },
         });
         
         History.pop();
         
         return response.text || question; // Fallback to original question if transformation fails
    } catch (error) {
        console.error('Query transformation error:', error);
        History.pop(); // Ensure we remove the added item even on error
        return question; // Return original question if transformation fails
    }
}

async function chatting(question){
    try {

        //convert the question into complete question with chat history

        const queries = await transformQuery(question);

        //convert the query into vector
        //use previous embedding model googleGenAI
        const embeddings = new GoogleGenerativeAIEmbeddings({
            apiKey: process.env.GEMINI_API_KEY,
            model: 'text-embedding-004',
            });
         
        // Validate queries before embedding
        if (!queries || typeof queries !== 'string' || queries.trim().length === 0) {
            throw new Error('Invalid query: query transformation returned empty or invalid result');
        }
        
        const queryVector = await embeddings.embedQuery(queries);
        //query vector 

        //now search the relevant vector in vector db
        const pinecone = new Pinecone({
            apiKey: process.env.PINECONE_API_KEY,
        });
        const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);
        const searchResults = await pineconeIndex.query({
            topK: 10,
            vector: queryVector,
            includeMetadata: true,
            });
        
        //console.log(searchResults);
        // top 10 documents: 10 metadata and we want only text

        const context = searchResults.matches
                       .map(match => match.metadata?.text)
                       .filter(text => text) // Remove any undefined values
                       .join("\n\n---\n\n");
        //NOW THIS is the context for the LLM 

        //Gemini Model - keep context in system instruction, question separate
        History.push({
            role:'user',
            parts:[{text:queries}]
            })
        
            
        const response = await ai.models.generateContent({
            model: "gemini-2.5-flash",
            contents: History,
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
        
        History.push({
            role:'model',
            parts:[{text:responseText}]
            })

        console.log("\n");
        console.log(responseText);
    } catch (error) {
        console.error("Error in chatting function:", error.message);
    }
}

async function main(){
   while (true) {
      const userProblem = readlineSync.question("Ask me anything--> ");
      if (userProblem.toLowerCase() === 'exit' || userProblem.toLowerCase() === 'quit') {
         console.log("Goodbye!");
         break;
      }
      await chatting(userProblem);
   }
}

main();