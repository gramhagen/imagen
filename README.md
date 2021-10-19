# Imagen
Streamlit app for AI based image generation

# Setup


## Using Bicep
See [Getting Started](https://github.com/Azure/bicep/tree/main#get-started-with-bicep) guide at Azure Bicep repo

# Deployment
```bash
RG=<RESOURCE GROUP NAME>
LOCATION=<RESOURCE GROUP LOCATION>
az group create -n $RG -l $LOCATION
az deployment group create -f ./main.bicep -g $RG -p username=$USER -p adminPasswordOrKey=$(cat ~/.ssh/id_rsa.pub)
IP=$(az vm show -d -g $RG -n ImagenVM --query "publicIps" -o tsv)
```