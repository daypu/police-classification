import streamlit as st
from aligo import Aligo
import os
import platform

# 初始化Aligo实例
refresh_token = "1e46313914494ac5875bb7b69e0c0212"
ali = Aligo(refresh_token=refresh_token)

# 获取用户桌面路径函数
def get_desktop_path():
    home = os.path.expanduser("~")
    if platform.system() == "Windows":
        return os.path.join(home, "Desktop")
    elif platform.system() == "Darwin":  # macOS
        return os.path.join(home, "Desktop")
    else:  # Linux and other OS
        return os.path.join(home, "Desktop")

# 下载文件或文件夹函数
def down_file_or_folder(remote_path, local_folder, is_file=False):
    file = ali.get_file_by_path(remote_path) if is_file else ali.get_folder_by_path(remote_path)
    if is_file:
        ali.download_file(file_id=file.file_id, local_folder=local_folder)
    else:
        ali.download_folder(folder_file_id=file.file_id, local_folder=local_folder)

# Streamlit应用界面
def main():
    st.title("下载文件示例")

    # 获取桌面路径
    desktop_path = get_desktop_path()
    st.write(f"文件将下载到: {desktop_path}")

    # 下载按钮
    if st.button("下载模型权重"):
        remote_path = 'Files/pth/model.pth'  # 替换为你的远程目录路径
        try:
            down_file_or_folder(remote_path, desktop_path, is_file=True)  # 这里假设只下载单个文件
            st.success("模型权重文件下载完成！")
        except Exception as e:
            st.error(f"下载失败: {e}")

        # 显示下载的文件路径
        local_file_path = os.path.join(desktop_path, "model.pth")  # 替换为你实际的文件路径
        st.write(f"模型权重文件已保存到: {local_file_path}")

if __name__ == "__main__":
    main()
