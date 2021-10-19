@description('Size of vm')
param vmSize string = 'Standard_D2s_v3'
// param vmSize string = 'Standard_NC6s_v3'

@description('Username for the Virtual Machine.')
param username string

@description('The base URI where artifacts required by this template are located. When the template is deployed using the accompanying scripts, a private location in the subscription will be used and this value will be automatically generated.')
//param artifactsLocation string = deployment().properties.templateLink.uri
param artifactsLocation string = 'https://raw.githubusercontent.com/gramhagen/imagen/main/'

@description('Location for all resources.')
param location string = resourceGroup().location

@description('SSH Key or password for the Virtual Machine. SSH key is required.')
@secure()
param sshPublicKey string

@description('Size of OS Disk in GB.')
param osDiskSize int = 128

var prefix = 'Imagen1'
var publicIPAddressName = '${prefix}PublicIp'
var publicIPAddressType = 'Dynamic'
var vnetName = '${prefix}VNet'
var vnetAddressPrefix = '10.0.0.0/16'
var subnetName = '${prefix}Subnet'
var subnetPrefix = '10.0.0.0/24'
var nicName = '${prefix}Nic'
var imagePublisher = 'canonical'
var imageOffer = '0001-com-ubuntu-server-focal'
var ubuntuOSVersion = '20_04-lts-gen2'
var vmName = '${prefix}VM'

resource publicIPAddress 'Microsoft.Network/publicIPAddresses@2021-02-01' = {
  name: publicIPAddressName
  location: location
  properties: {
    publicIPAllocationMethod: publicIPAddressType
  }
}

resource vnet 'Microsoft.Network/virtualNetworks@2021-02-01' = {
  name: vnetName
  location: location
  properties: {
    addressSpace: {
      addressPrefixes: [
        vnetAddressPrefix
      ]
    }
    subnets: [
      {
        name: subnetName
        properties: {
          addressPrefix: subnetPrefix
        }
      }
    ]
  }
}

resource nic 'Microsoft.Network/networkInterfaces@2021-02-01' = {
  name: nicName
  location: location
  properties: {
    ipConfigurations: [
      {
        name: 'ipconfig1'
        properties: {
          privateIPAllocationMethod: 'Dynamic'
          publicIPAddress: {
            id: publicIPAddress.id
          }
          subnet: {
            id: resourceId('Microsoft.Network/virtualNetworks/subnets', vnetName, subnetName)
          }
        }
      }
    ]
  }
  dependsOn: [
    vnet
  ]
}

resource vm 'Microsoft.Compute/virtualMachines@2021-03-01' = {
  name: vmName
  location: location
  properties: {
    hardwareProfile: {
      vmSize: vmSize
    }
    osProfile: {
      computerName: vmName
      adminUsername: username
      adminPassword: sshPublicKey
      linuxConfiguration: {
        disablePasswordAuthentication: true
        ssh: {
          publicKeys: [
            {
              path: '/home/${username}/.ssh/authorized_keys'
              keyData: sshPublicKey
            }
          ]
        }
      }
    }
    storageProfile: {
      imageReference: {
        publisher: imagePublisher
        offer: imageOffer
        sku: ubuntuOSVersion
        version: 'latest'
      }
      osDisk: {
        caching: 'ReadWrite'
        createOption: 'FromImage'
        diskSizeGB: osDiskSize
      }
    }
    networkProfile: {
      networkInterfaces: [
        {
          id: nic.id
        }
      ]
    }
  }
}

resource vm_customScript 'Microsoft.Compute/virtualMachines/extensions@2021-04-01' = {
  parent: vm
  name: 'customScript'
  location: location
  properties: {
    publisher: 'Microsoft.Azure.Extensions'
    type: 'CustomScript'
    typeHandlerVersion: '2.0'
    autoUpgradeMinorVersion: true
    settings: {
      fileUris: [
        uri(artifactsLocation, 'deployment/scripts/setup.sh')
      ]
      commandToExecute: 'bash deploy.sh'
    }
  }
}
