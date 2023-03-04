import numpy as np
import cv2
import os
from keras.models import load_model
import matplotlib.pyplot as plt
import ray

# ray.init()

def plot_tangle(best_frame, rgb_img0):
    tangle_x1 = int(best_frame[0] - longth * best_frame[2] / 2)
    tangle_x2 = int(best_frame[0] + longth * best_frame[2] / 2)
    tangle_y1 = int(best_frame[1] - width * best_frame[2] / 2)
    tangle_y2 = int(best_frame[1] + width * best_frame[2] / 2)

    M = np.eye(3)
    M[:2, :] = cv2.getRotationMatrix2D(best_frame[:2], best_frame[3], 1)

    p1 = M.dot(np.array([[tangle_x1, tangle_y1, 1]]).T).T
    p2 = M.dot(np.array([[tangle_x1, tangle_y2, 1]]).T).T
    p3 = M.dot(np.array([[tangle_x2, tangle_y1, 1]]).T).T
    p4 = M.dot(np.array([[tangle_x2, tangle_y2, 1]]).T).T

    p_1 = tuple(int(i/p1[0, 2]) for i in p1[0, :2])
    p_2 = tuple(int(i/p2[0, 2]) for i in p2[0, :2])
    p_3 = tuple(int(i/p3[0, 2]) for i in p3[0, :2])
    p_4 = tuple(int(i/p4[0, 2]) for i in p4[0, :2])

    img_plot = rgb_img0.copy()

    cv2.line(img_plot, p_1, p_2, color=(255, 0, 0), thickness=1)
    cv2.line(img_plot, p_1, p_3, color=(255, 0, 0), thickness=1)
    cv2.line(img_plot, p_4, p_2, color=(255, 0, 0), thickness=1)
    cv2.line(img_plot, p_4, p_3, color=(255, 0, 0), thickness=1)

    return img_plot




# @ray.remote
def Cnn2_3(center, rgb_img, depth_img, cnn2, cnn3):
    save_img = []
    save_frame = []
    for r in range(0, 180, 15):
        # print(r)
        M = cv2.getRotationMatrix2D(center, r, 1)
        img = cv2.warpAffine(rgb_img, M, (rgb_img.shape[1], rgb_img.shape[0]), borderValue=(255, 255, 255))[np.newaxis, :, :, :] / 255
        img_cnn2 = img[:, int(center[0] - width/2):int(center[0] + width/2), int(center[1] - longth/2):int(center[1] + longth/2), :]
        # plt.subplot(1, 2, 1)
        # plt.imshow(rgb_img)
        # plt.subplot(1, 2, 2)
        # plt.imshow(img_cnn2[0])
        # plt.show()
        # print("1")

        img = cv2.warpAffine(depth_img, M, (rgb_img.shape[1], rgb_img.shape[0]), borderValue=(255, 255, 255))[np.newaxis, :, :,
                   np.newaxis]
        img_cnn3 = img[:, int(center[0] - width/2):int(center[0] + width/2), int(center[1] - longth/2):int(center[1] + longth/2), :]

        prediction_cnn2 = cnn2.predict(img_cnn2, verbose=0)[0, 0]
        prediction_cnn3 = cnn3.predict(img_cnn3, verbose=0)[0, 0]

        # plt.imshow(img_cnn2[0])
        # plt.show()
        # print("the cnn2 {}, the cnn3 {}".format(prediction_cnn2, prediction_cnn3))

        if prediction_cnn2 > 0.5 and prediction_cnn3 > 0.5:
            save_img.append(img_cnn2)
            save_frame.append((center[0] * cutdown, center[1] * cutdown, cutdown, r))
    # print(len(save_img))
    return save_img, save_frame


if __name__=="__main__":
    file_path = "/home/mxs/Downloads/source_code_of_paper_1/01-Source_Code_and_Result_for_Testing/image-wise/data-1"

    n = 16 # 滑动图像框大小
    img_borde = np.array([[120, 60], [60, 120]])

    width = 16
    longth = 32

    save_path = "/home/mxs/python_project/source_code_of_paper_2-master/GRDNs/result/"


    rgb_date = []
    depth_date = []
    for i in range(100, 501):
        path_rgb = os.path.join(file_path, "pcd{:0>4d}r.png".format(i))
        path_depth = os.path.join(file_path, "pcd{:0>4d}d.tiff".format(i))
        # img = cv2.imread(path_rgb)[:, :, ::-1]
        rgb_date.append(cv2.imread(path_rgb)[:, :, ::-1])
        depth_img = cv2.imread(path_depth, -1)
        depth_date.append(depth_img[:, :, np.newaxis])


    cnn1 = load_model("CNN-1.h5")
    cnn2 = load_model("CNN-2.h5")
    cnn3 = load_model("CNN-3.h5")   #depth
    cnn4 = load_model("CNN-4.h5")

    best_frame = []
    img_i = 100
    for rgb_img0, depth_img0 in zip(rgb_date, depth_date):
        print("the img is {}".format(img_i))
        depth_img0 = depth_img0 * 100
        save_img = []
        save_frame = []

        for cutdown in range(2, 5):
            rgb_img = rgb_img0[0:-1:cutdown, 0:-1:cutdown, :]
            depth_img = depth_img0[0:-1:cutdown, 0:-1:cutdown, :]

            img_size = rgb_img.shape
            center_list = []

            for i_width in range(np.int64(img_borde[0, 0] / cutdown), np.int64(img_size[0] - img_borde[0, 1] / cutdown - n), n):
                for i_longth in range(np.int64(img_borde[1, 0] / cutdown), np.int64(img_size[1] - img_borde[1, 1] / cutdown - n), n):
                    # print(i_width, i_longth)
                    # plt.imshow(rgb_img0)
                    # plt.show()
                    img_cnn1 = np.array([rgb_img[i_width: i_width + n, i_longth: i_longth + n, :] / 255])
                    prediction_cnn1 = cnn1.predict(img_cnn1, verbose=0)[0, 0]
                    # print(prediction_cnn1)
                    if prediction_cnn1 > 0.5:
                        # plt.imshow(img_cnn1[0])
                        # plt.show()
                        center_list.append((i_width + 4, i_longth + 4))
                        center_list.append((i_width + 4, i_longth + 12))
                        center_list.append((i_width + 12, i_longth + 4))
                        center_list.append((i_width + 12, i_longth + 12))



            all_save = []
            for center in center_list:
                # for r in range(0, 180, 15):
                #     M = cv2.getRotationMatrix2D(center, r, 1)
                #     img_cnn2 = cv2.warpAffine(rgb_img, M, (longth, width), borderValue=(255, 255, 255))[np.newaxis, :, :, :] / 255
                #     img_cnn3 = cv2.warpAffine(depth_img, M, (longth, width), borderValue=(255, 255, 255))[np.newaxis, :, :, np.newaxis]
                #
                #     prediction_cnn2 = cnn2.predict(img_cnn2, verbose=0)[0, 0]
                #     prediction_cnn3 = cnn3.predict(img_cnn3, verbose=0)[0, 0]
                #
                #     if prediction_cnn2 > 0.5 and prediction_cnn3 > 0.5:
                #         save_img.append(img_cnn2)
                #         save_frame.append((np.array([center])*cutdown, cutdown, r))
                all_save.append(Cnn2_3(center, rgb_img, depth_img, cnn2, cnn3))
            # all_save = ray.get(all_save)

            for save_img1, save_frame1 in all_save:
                save_img += save_img1
                save_frame += save_frame1


        if len(save_img) == 0:
            print("Error, no best")
            # best_frame.append([])
            img_i += 1
            continue
        elif len(save_img) == 1:
            print("Only one")
            np.save(save_path + 'frame/' + 'pcd{:0>4d}.npy'.format(img_i), best_frame[-1])

            # best_frame = [(150, 150, 2, 0)]
            print(save_frame[-1])
            img = plot_tangle(save_frame[-1], rgb_img0)
            cv2.imwrite(save_path + 'image/' + 'pcd{:0>4d}.png'.format(img_i), img)
            img_i += 1
            # plt.subplot(1, 2, 1)
            # plt.imshow(rgb_img0)
            # plt.subplot(1, 2, 2)
            # plt.imshow(save_img[-1])
            # plt.show()
            # best_frame.append(save_frame[-1])
            continue

        perfance = []

        for img in save_img:

            prediction_cnn4 = cnn4.predict(img, verbose=0)[0, 0]


            perfance.append(prediction_cnn4)

        # perfance = np.array(perfance)

        best = np.argsort(perfance)[::-1][:3]
        mean_coordinates = np.mean(np.array(save_frame)[:, :2], axis=0)

        coordinates = []

        for i in range(3):
            coordinates.append(np.sum(np.abs(np.array(save_frame)[best[i], :2] - mean_coordinates), axis=0))


        best_frame.append(save_frame[best[np.argmin(coordinates)]])

        np.save(save_path+'frame/'+'pcd{:0>4d}.npy'.format(img_i), best_frame[-1])

        # best_frame = [(150, 150, 2, 0)]
        print(best_frame[-1])
        img = plot_tangle(best_frame[-1], rgb_img0)
        cv2.imwrite(save_path+'image/'+'pcd{:0>4d}.png'.format(img_i), img[:, :, ::-1])
        img_i += 1
        # plt.imshow(img)
        # plt.show()



        # print("1")