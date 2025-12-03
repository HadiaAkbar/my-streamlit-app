Set WshShell = CreateObject("WScript.Shell")
Set objFSO = CreateObject("Scripting.FileSystemObject")

projectPath = "C:\Users\LTC\OneDrive\Desktop\MH[FG]"
batchFile = projectPath & "\run_mhfg.bat"

' Create batch file if it doesn't exist
If Not objFSO.FileExists(batchFile) Then
    Set objFile = objFSO.CreateTextFile(batchFile, True)
    objFile.WriteLine "@echo off"
    objFile.WriteLine "cd /d " & projectPath
    objFile.WriteLine "call venv\Scripts\activate.bat"
    objFile.WriteLine "streamlit run app.py"
    objFile.WriteLine "pause"
    objFile.Close
End If

' Run the batch file
WshShell.CurrentDirectory = projectPath
WshShell.Run "cmd.exe /k """ & batchFile & """", 1, False