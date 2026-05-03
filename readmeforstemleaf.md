  1. 训练命令                                                                                                                                                                                
                                                                                                                                                                                             
  python tools/train.py configs/myplants.yaml --work_dir exps/myplants_1024                                                                                                                                               
                                                                                                                                                                                             
  2. 测试命令                                                                                                                                                                                
                                                                                                                                                                                             
  python tools/test.py configs/myplants.yaml checkpoints/模型名.pth                                                                                                                          
                                                                                                                                                                                             
  3. 保存测试结果为PLY格式                                                                                                                                                                   
                                                                                                                                                                                             
  python tools/save_segmented_ply.py \                                                                                                                                                       
      configs/myplants.yaml \                                                                                                                                                                
      checkpoints/模型名.pth \                                                                                                                                                               
      --output-dir ./output/segmented_ply                                                                                                                                                    
                                                                                                                                                                                             
  4. 可视化训练过程（图表）                                                                                                                                                                  
                                                                                                                                                                                             
  tensorboard --logdir exps/myplants --port 6006                                                                                                                                             
  然后浏览器打开：http://localhost:6006                                                                                                                                                      
                                                                                                                                                                                             
  ---                                                                                                                                                                                        
  说明：                                                                                                                                                                                     
  - 训练输出在 exps/myplants/ 目录                                                                                                                                                           
  - 模型检查点文件如 epoch_256.pth                                                                                                                                                           
  - PLY文件会按场景保存，格式和raw目录一样（stem.ply, leaf_XXX_1.ply等）                                                                                                                     
  - TensorBoard会显示loss曲线、学习率、评估指标等   
