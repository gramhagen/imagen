# Imagen
Streamlit app for AI based image generation

# Setup

## Using Bicep
See [Getting Started](https://github.com/Azure/bicep/tree/main#get-started-with-bicep) guide at Azure Bicep repo

# Deployment
RG=<RESOURCE GROUP NAME>
LOCATION=<RESOURCE GROUP LOCATION>
az group create -n $RG -l $LOCATION
az deployment group create -f ./main.bicep -g $RG