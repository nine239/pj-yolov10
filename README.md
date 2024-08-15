generate_images.py : files python chạy camera, bấm s để chụp 1 frame ảnh và lưu vào folder images_for_train.



coordination_image.py,  coordination_video.py  : xác định vật thể vẽ bounding box 1 Rectangles 1, Overlap Area... Total Square, Percentage
                        2 Rectangles 1,2,3, Overlap Area... Total Square, Percentage Với mỗi framew thì đẩy 1 dòng vào dataframe hoặc đẩy 1 dòng ra file csv với ảnh hoặc video.

                        
imagestest.py và vdtest.py : chỉ là 2 file test code của coordination_image.py,  coordination_video.py



custom_data : ghi đường dẫn của folder custom_dataset
