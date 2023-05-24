## export your open api key.
export OPENAI_API_KEY=sk-x...

## ingest the data
./ingest.sh

## to start the web interface
make start

## to start the API endpoint
uvicorn srv:app --reload

## example curl request 
curl -X POST -H "Content-Type: application/json" -d '{"question": "What is triggermesh?"}' http://localhost:8000/chat
