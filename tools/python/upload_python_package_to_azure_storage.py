#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import argparse
from azure.identity import ManagedIdentityCredential
from azure.storage.blob import BlobServiceClient, ContentSettings


def upload_whl(python_wheel_path, account_name, managed_identity_client_id, container_name):
    managed_id_credential = ManagedIdentityCredential(client_id=managed_identity_client_id)

    blob_service_client = BlobServiceClient(f"https://{account_name}.blob.core.windows.net",
                                            credential=managed_id_credential)

    blob_name = os.path.basename(python_wheel_path)
    blob_client = blob_service_client.get_blob_client(container_name, blob_name)
    with open(python_wheel_path, "rb") as blob:
        blob_client.upload_blob(blob, blob_type="BlockBlob", overwrite=True)

    html_blob_name = 'torch_ort_nightly.html'
    html_blob_client = blob_service_client.get_blob_client(container_name, html_blob_name)
    lines = html_blob_client.download_blob().content_as_text().splitlines()

    new_line = '<a href="{blobname}">{blobname}</a><br>'.format(blobname=blob_name)
    lines.append(new_line)
    lines.sort()
    html_blob = "\n".join(lines)
    content_settings = ContentSettings(content_type='text/html')
    html_blob_client.upload_blob(html_blob, blob_type="BlockBlob", content_settings=content_settings, overwrite=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload python whl to azure storage.")

    parser.add_argument("--python_wheel_path", type=str, help="path to python wheel")
    parser.add_argument("--account_name", type=str, help="name of the Azure storage account that is used to store package files")
    parser.add_argument("--managed_identity_client_id", type=str, help="Managed Identity client id to use for authentication")
    parser.add_argument("--container_name", type=str, help="the container name within the storage account for the packages")

    args = parser.parse_args()

    upload_whl(args.python_wheel_path, args.account_name, args.managed_identity_client_id, args.container_name)
