generate-vector-embedings:
	@pdm run python src/embeding/embed.py 
	$(call teardown)

local-setup:
	@cp -f .env.template .env
	@pdm config python.use_venv true
	@pdm install --no-self

start-query-genration:
	@pdm run python src/app.py


format:
	@pdm run black .

start:
	@streamlit run src/app.py

local-setup:
	docker run -d --name llm-demo -p 5432:5432 sarveshdev92/vector-db-image:latest
	@pdm run python src/embeding/embed.py
