import SimpleITK as sitk
p = '/tmp/ts_test/kidney_left.nii.gz'
try:
    img = sitk.ReadImage(p)
    vol = sitk.GetArrayFromImage(img).sum()
    print("Success! Volume:", vol)
except Exception as e:
    print("Error:", e)
