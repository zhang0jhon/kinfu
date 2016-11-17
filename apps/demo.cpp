#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/viz/vizcore.hpp>
#include <opencv2/imgcodecs.hpp>
#include <kfusion/kinfu.hpp>
#include <io/capture.hpp>
#include <fstream>
#include <string>
#include <vector>
// #include <pcl/io/pcd_io.h>

using namespace kfusion;
// using namespace pcl;
// using namespace cv;

struct Coordinate
{
    int x;
    int y;
};

struct KinFuApp
{
    static void KeyboardCallback(const cv::viz::KeyboardEvent& event, void* pthis)
    {
        KinFuApp& kinfu = *static_cast<KinFuApp*>(pthis);

        if(event.action != cv::viz::KeyboardEvent::KEY_DOWN)
            return;

        if(event.code == 't' || event.code == 'T')
            kinfu.take_cloud(*kinfu.kinfu_);

        if(event.code == 'i' || event.code == 'I')
            kinfu.iteractive_mode_ = !kinfu.iteractive_mode_;
    }

    KinFuApp(OpenNISource& source) : exit_ (false),  iteractive_mode_(false), capture_ (source)
    {
        KinFuParams params = KinFuParams::default_params();
        kinfu_ = KinFu::Ptr( new KinFu(params) );

        capture_.setRegistration(true);

        cv::viz::WCube cube(cv::Vec3d::all(0), cv::Vec3d(params.volume_size), true, cv::viz::Color::apricot());
        viz.showWidget("cube", cube, params.volume_pose);
        viz.showWidget("coor", cv::viz::WCoordinateSystem(0.1));
        viz.registerKeyboardCallback(KeyboardCallback, this);
    }

    void show_depth(const cv::Mat& depth)
    {
        cv::Mat display;
        //cv::normalize(depth, display, 0, 255, cv::NORM_MINMAX, CV_8U);
        depth.convertTo(display, CV_8U, 255.0/4000);
        cv::imshow("Depth", display);
    }

    void show_raycasted(KinFu& kinfu)
    {
        const int mode = 3;
        if (iteractive_mode_)
            kinfu.renderImage(view_device_, viz.getViewerPose(), mode);
        else
            kinfu.renderImage(view_device_, mode);

        view_host_.create(view_device_.rows(), view_device_.cols(), CV_8UC4);
        view_device_.download(view_host_.ptr<void>(), view_host_.step);
        cv::imshow("Scene", view_host_);
    }

    void savePointClouds(cv::Mat& pointClouds, const char* filename)
    {
        const double max_z = 1.0e4;
        const double r=0.07;

        try
        {
            FILE* fp = fopen(filename, "a+");

            double z=0;
        
             for(int y = 0; y < pointClouds.rows; y++)
            {
                for(int x = 0; x < pointClouds.cols; x++)
                {
                    cv::Vec3f point = pointClouds.at<cv::Vec3f>(y, x);
                    if(point[2]<1.500 && point[2]>0.5 ){
                        if(pow(point[0],2)+pow(point[1],2)<pow(0.005,2))
                            z=point[2];
                    }
                }
            }
            if(z==0){
                std::cout<<"here here here~"<<std::endl;
                 for(int y = 0; y < pointClouds.rows; y++)
                {
                    for(int x = 0; x < pointClouds.cols; x++)
                    {
                        cv::Vec3f point = pointClouds.at<cv::Vec3f>(y, x);
                        if(point[2]<1.500 && point[2]>0.5 ){
                            if(pow(point[0],2)+pow(point[1],2)<pow(0.015,2))
                                z=point[2];
                        }
                    }
                }
            }
            if(z==0){
                std::cout<<"here here here~"<<std::endl;
                 for(int y = 0; y < pointClouds.rows; y++)
                {
                    for(int x = 0; x < pointClouds.cols; x++)
                    {
                        cv::Vec3f point = pointClouds.at<cv::Vec3f>(y, x);
                        if(point[2]<1.500 && point[2]>0.5 ){
                            if(pow(point[0],2)+pow(point[1],2)<pow(0.025,2))
                                z=point[2];
                        }
                    }
                }
            }
            if(z==0){
                z=0.65;
                std::ofstream file0;
                file0.open("find_no_z.txt", std::ios::out | std::ios::binary | std::ios::app);
                file0<<filename<<std::endl;
                file0.close();
            }

            
            std::cout<<"z: "<<z<<std::endl;
             for(int y = 0; y < pointClouds.rows; y++)
            {
                for(int x = 0; x < pointClouds.cols; x++)
                {
                    cv::Vec3f point = pointClouds.at<cv::Vec3f>(y, x);
                    if(point[2]<1.500 && point[2]>0.5 ){
                        if(z!=0){
                            double dis=pow(point[0],2)+pow(point[1],2)+pow(z-point[2],2);

                            if(dis<pow(r,2))
                                fprintf(fp, "v %f %f %f\n", point[0], point[1], point[2]);

                        }
                        else
                            fprintf(fp, "v %f %f %f\n", point[0], point[1], point[2]);
                // printf("%f %f %f\n", point[0], point[1], point[2]);
                    }
            
                }
            }
        
            fclose(fp);
        }
        catch (std::exception* e)
        {
            printf("Failed to save point clouds. Error: %s \n\n", e->what());
        }
    }
    void take_cloud(KinFu& kinfu)
    {
        cuda::DeviceArray<Point> cloud = kinfu.tsdf().fetchCloud(cloud_buffer);
        cv::Mat cloud_host(1, (int)cloud.size(), CV_32FC4);
        cloud.download(cloud_host.ptr<Point>());
        viz.showWidget("cloud", cv::viz::WCloud(cloud_host));
        // savePointClouds(cloud_host,"a.obj");
        //viz.showWidget("cloud", cv::viz::WPaintedCloud(cloud_host));
    }

    
    bool execute(const char * filename,const char *fobj)
    {
        KinFu& kinfu = *kinfu_;
        cv::Mat depth, image;
        double time_ms = 0;
        bool has_image = false;
        std::vector<struct Coordinate> cor;

        std::string name(filename);
        std::string::size_type point = name.rfind("kinect-2/");
        std::string path = name.replace(0,point+8,"/home/zhang_jhon/matlab/data/kinect-2");
        point = path.rfind("DEPTH.RAW");
        std::string txtpath = path.replace(point,12,"INFRARED.txt");
        std::ifstream txt;
        txt.open(txtpath.c_str(), std::ios::in | std::ios::binary);
        if(!txt)
            return 0;
        std::string line;
        while(getline(txt,line))
        {
            struct Coordinate cor1;
            std::string::size_type index1 = line.find_first_of(":");
            std::string::size_type index2 = line.find(" ");
            std::string::size_type index3 = line.find_last_of(":");
            std::string s1 = line.substr(index1+1, index2);
            std::string s2 = line.substr(index3+1);
            cor1.x = atoi(s1.c_str());
            cor1.y = atoi(s2.c_str());
            // std::cout<<cor1.x<<"  "<<cor1.y<<std::endl;
            cor.push_back(cor1);
        }
        txt.close();

        int x,y;
        if (cor[2].x==0 && cor[2].y==0){
            std::cout << "without nosetip!\n" <<  std::endl;
            if(cor[0].x==0 && cor[0].y==0&&cor[1].x==0 && cor[3].y==0){
                    x=(cor[0].x+cor[1].x)/2;
                    y=(cor[0].y+cor[3].y)/2;
            }
            else
                {
                        for (int j=0;j<5;j++){
                            if(cor[j].x!=0&&cor[j].y!=0){
                                        x=cor[j].x;
                                        y=cor[j].y;
                                }
                            }
                            if(x==0&&y==0){
                                return 0;
                            }
                }
        }
        else{
            x=cor[2].x;
            y=cor[2].y;
        }
        std::cout<<"x: "<<x<<"  y: "<<y<<std::endl;
        int w=150;
        int h=150;
        cv::Mat m_depth16u(424, 512, CV_16UC1);
        cv::Mat m_depth(w, h, CV_16UC1);
        std::ifstream filein;
        filein.open(filename, std::ios::in | std::ios::binary);
        char *ptr = (char *)m_depth16u.data;
        // std::string s();
        for (int i = 0; !exit_ && !viz.wasStopped()&& !filein.eof()&& i<60; ++i)//&& !filein.eof()
        {
            bool has_frame = capture_.grab(depth, image);
            if (!has_frame)
            {
                std::ofstream file1;
                file1.open("fail.txt", std::ios::out | std::ios::binary | std::ios::app);
                file1<<filename<<std::endl;
                file1.close();
                return std::cout << "Can't grab" << std::endl, false;
            }

            filein.read(ptr, 424 * 512 * 2);
            cv::Mat row = m_depth16u.rowRange(y-h/2,y+h/2).clone();
            m_depth = row.colRange(x-w/2, x+w/2).clone();
            // cv::resize(m_depth,m_depth,cv::Size(150,150),0,0,3); 
            cv::imwrite("depth.png", m_depth);
            depth=cv::imread("depth.png",CV_LOAD_IMAGE_UNCHANGED);

            depth_device_.upload(depth.data, depth.step, depth.rows, depth.cols);

            {
                SampledScopeTime fps(time_ms); (void)fps;
                has_image = kinfu(depth_device_);
            }

        }

        
        cuda::DeviceArray<Point> cloud = kinfu.tsdf().fetchCloud(cloud_buffer);
        cv::Mat cloud_host(1, (int)cloud.size(), CV_32FC4);
        cloud.download(cloud_host.ptr<Point>());
        savePointClouds(cloud_host,fobj);
        std::cout<<"save done!"<<std::endl;
        
        filein.close();
        return true;
    }

    bool exit_, iteractive_mode_;
    OpenNISource& capture_;
    KinFu::Ptr kinfu_;
    cv::viz::Viz3d viz;

    cv::Mat view_host_;
    cuda::Image view_device_;
    cuda::Depth depth_device_;
    cuda::DeviceArray<Point> cloud_buffer;
};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main (int argc, char* argv[])
{
    int device = 0;
    cuda::setDevice (device);
    cuda::printShortCudaDeviceInfo (device);

    if(cuda::checkIfPreFermiGPU(device))
        return std::cout << std::endl << "Kinfu is not supported for pre-Fermi GPU architectures, and not built for them by default. Exiting..." << std::endl, 1;

    OpenNISource capture;
    capture.open (0);
    
    // capture.open("/home/zhang_jhon/kinfu_remake/apps/demo.oni");
    //capture.open("d:/onis/white1.oni");
    //capture.open("/media/Main/onis/20111013-224932.oni");
    //capture.open("20111013-225218.oni");
    //capture.open("d:/onis/20111013-224551.oni");
    //capture.open("d:/onis/20111013-224719.oni");

    KinFuApp app (capture);
    // std::cout<<argv[1]<<std::endl;
    // std::ifstream fname("a.txt",std::ios::in | std::ios::binary);
    // std::string s;
    // while(getline(fname,s))
    // {
        const char *f =argv[1];
        std::string s1=argv[1];
        std::string::size_type point = s1.rfind("kinect-2/");
        std::string obj= s1.replace(0,point+8,"/home/zhang_jhon/matlab/data/kinect-2");
        point = obj.rfind("RAW");
        obj = obj.replace(point,3,"obj");
        
        const char * f1=obj.c_str();
        printf("%s\n",f);
        try { 
            app.execute (f,f1); 
        }
        catch (const std::bad_alloc& /*e*/) { std::cout << "Bad alloc" << std::endl; }
        catch (const std::exception& /*e*/) { std::cout << "Exception" << std::endl; }

    // }
    // fname.close();

    return 0;
}
