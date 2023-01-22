initialize_git:
	@echo "Initializing git..."
	git init 
	
create: 
	@echo "Installing..."
	conda env create -f environment.yml

activate:
	@echo "Activating virtual environment"
	conda activate pysparkenv

setup: initialize_git create