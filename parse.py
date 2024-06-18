import aligo
from aligo import Aligo

refresh_token = "1e46313914494ac5875bb7b69e0c0212"
ali = Aligo(refresh_token=refresh_token)

def down_file_or_folder(remote_path, local_folder, is_file=False):
    file = ali.get_file_by_path(remote_path) if is_file else ali.get_folder_by_path(remote_path)
    ali.download_file(file_id=file.file_id, local_folder=local_folder) if is_file else ali.download_folder(folder_file_id=file.file_id, local_folder=local_folder)

def download_file():
    remote_path = 'Files/pth/'  # Replace with your specific remote directory path
    local_folder = './'     # Replace with your local folder path where you want to download
    
    down_file_or_folder(remote_path, local_folder)
    path = "/pth/model.pth"

    return path
if __name__ == "__main__":
    download_file()