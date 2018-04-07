#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/console/print.h>
#include <pcl/console/parse.h>
// #include <pcl/console/time.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <iostream>
#include <fstream>
#include <json/json.h>
// #include <json/writer.h>

using namespace std;

bool loadCloud (const string &filename, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud){
  ifstream fs;
  fs.open (filename.c_str (), ios::binary);
  if (!fs.is_open () || fs.fail ()){
    PCL_ERROR ("Could not open file '%s'! Error : %s\n", filename.c_str (), strerror (errno)); 
    fs.close ();
    return (false);
  }
  
  string line;
  vector<string> st;

  while (!fs.eof ()){
    getline (fs, line);
    // Ignore empty lines
    if (line == "")
      continue;

    // Tokenize the line
    boost::trim (line);
    boost::split (st, line, boost::is_any_of ("\t\r "), boost::token_compress_on);

    if (st.size () != 3)
      continue;

    cloud->push_back (pcl::PointXYZ (float (atof (st[0].c_str ())), float (atof (st[1].c_str ())), float (atof (st[2].c_str ()))));
  }
  fs.close ();

  cloud->width = uint32_t (cloud->size ()); 
  cloud->height = 1; 
  cloud->is_dense = true;
  return (true);
}

void viewCloud(const string viewername, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud){
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addPointCloud<pcl::PointXYZ> (cloud, viewername);
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, viewername);
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
  while (!viewer->wasStopped ()){
  	viewer->spinOnce (100);
  	boost::this_thread::sleep (boost::posix_time::microseconds (100000));
  }

  return;
}

bool pcl::visualization::PCLVisualizer::addCube (
  const Eigen::Vector3f &translation, const Eigen::Quaternionf &rotation,
  double width, double height, double depth,
  const std::string &id, int viewport)
{
  // Check to see if this ID entry already exists (has it been already added to the visualizer?)
  ShapeActorMap::iterator am_it = shape_actor_map_->find (id);
  if (am_it != shape_actor_map_->end ())
  {
    pcl::console::print_warn (stderr, "[addCube] A shape with id <%s> already exists! Please choose a different id and retry.\n", id.c_str ());
    return (false);
  }

  vtkSmartPointer<vtkDataSet> data = createCube (translation, rotation, width, height, depth);

  // Create an Actor
  vtkSmartPointer<vtkLODActor> actor;
  createActorFromVTKDataSet (data, actor);
  actor->GetProperty ()->SetRepresentationToWireframe ();
  addActorToRenderer (actor, viewport);

  // Save the pointer/ID pair to the global actor map
  (*shape_actor_map_)[id] = actor;
  return (true);
}

void drawBoundingBox(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, Json::Value& vec){
    // compute principal direction
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cloud, centroid);
    Eigen::Matrix3f covariance;
    computeCovarianceMatrixNormalized(*cloud, centroid, covariance);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
    Eigen::Matrix3f eigDx = eigen_solver.eigenvectors();
    eigDx.col(2) = eigDx.col(0).cross(eigDx.col(1));

    // move the points to the that reference frame
    Eigen::Matrix4f p2w(Eigen::Matrix4f::Identity());
    p2w.block<3,3>(0,0) = eigDx.transpose();
    p2w.block<3,1>(0,3) = -1.f * (p2w.block<3,3>(0,0) * centroid.head<3>());
    pcl::PointCloud<pcl::PointXYZ> cPoints;
    pcl::transformPointCloud(*cloud, cPoints, p2w);

    pcl::PointXYZ min_pt, max_pt;
    pcl::getMinMax3D(cPoints, min_pt, max_pt);
    const Eigen::Vector3f mean_diag = 0.5f*(max_pt.getVector3fMap() + min_pt.getVector3fMap());

    // final transform
    const Eigen::Quaternionf qfinal(eigDx);
    const Eigen::Vector3f tfinal = eigDx*mean_diag + centroid.head<3>();

    // draw the cloud and the box
    // pcl::visualization::PCLVisualizer viewer;
    // viewer.addPointCloud(cloud);
    // viewer.addCube(tfinal, qfinal, max_pt.x - min_pt.x, max_pt.y - min_pt.y, max_pt.z - min_pt.z);
    // viewer.spin();

    // Eigen::IOFormat fmt(3, 0, "\t", " ", "", "");
    // cout<<tfinal.format(fmt)<<endl;
    // cout<<"scalar: "<<qfinal.w()<<endl;
    // cout<<"vector: "<<qfinal.vec()<<endl;
    // // output to json files
    Json::Value BoundingBox;

    BoundingBox["Translation"]["x"] = tfinal.x();
    BoundingBox["Translation"]["y"] = tfinal.y();
    BoundingBox["Translation"]["z"] = tfinal.z();
    Eigen::AngleAxisf r(qfinal);
    BoundingBox["Rotation"]["angle"] = r.angle();
    BoundingBox["Rotation"]["x_axis"] = r.axis()[0];
    BoundingBox["Rotation"]["y_axis"] = r.axis()[1];
    BoundingBox["Rotation"]["z_axis"] = r.axis()[2];
    BoundingBox["Width"] = max_pt.x - min_pt.x;
    BoundingBox["Height"] = max_pt.y - min_pt.y;
    BoundingBox["Depth"] = max_pt.z - min_pt.z;

    vec.append(BoundingBox);

    // Json::StyledWriter styledWriter;
    // fd << styledWriter.write(BoundingBox);

    // cout<<BoundingBox<<endl;
}

void planar_segmentation(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered){
	// Create the segmentation object for the planar model and set all the parameters
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane (new pcl::PointCloud<pcl::PointXYZ> ());
  pcl::PCDWriter writer;
  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setMaxIterations (100);
  seg.setDistanceThreshold (0.02);

  int i=0, nr_points = (int) cloud_filtered->points.size ();
  while (cloud_filtered->points.size () > 0.3 * nr_points)
  {
    // Segment the largest planar component from the remaining cloud
    seg.setInputCloud (cloud_filtered);
    seg.segment (*inliers, *coefficients);
    if (inliers->indices.size () == 0){
      cout << "Could not estimate a planar model for the given dataset." << endl;
      break;
    }

    // Extract the planar inliers from the input cloud
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud (cloud_filtered);
    extract.setIndices (inliers);
    extract.setNegative (false);

    // Get the points associated with the planar surface
    extract.filter (*cloud_plane);
    cout << "PointCloud representing the planar component: " << cloud_plane->points.size () << " data points." << endl;

    // Remove the planar inliers, extract the rest
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_f (new pcl::PointCloud<pcl::PointXYZ>);
    extract.setNegative (true);
    extract.filter (*cloud_f);
    *cloud_filtered = *cloud_f;
  }

  return;
}

void nearestNeighbor(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered){
	// Creating the KdTree object for the search method of the extraction
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud (cloud_filtered);
  pcl::PCDWriter writer;

  // cout<<*cloud_filtered<<endl;
  // cout<<tree->indices_<<endl;
  vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance (1.0); // 2cm
  ec.setMinClusterSize (50);
  ec.setMaxClusterSize (1000);
  ec.setSearchMethod (tree);
  ec.setInputCloud (cloud_filtered);
  ec.extract (cluster_indices);

  int j = 0;
  // cout<<"nearest neighbor"<<endl;
  // cout<<"size:"<<cluster_indices.size()<<endl;
  // cout<<"begin:"<<*cluster_indices.begin()<<endl;
  // cout<<"end:"<<*cluster_indices.end()<<endl;
 
  Json::Value vec(Json::arrayValue);
  for (vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it){
    // cout<<"iteration_"<<j<<endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
    
    for (vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
      cloud_cluster->points.push_back (cloud_filtered->points[*pit]); //*

    cloud_cluster->width = cloud_cluster->points.size ();
    cloud_cluster->height = 1;
    cloud_cluster->is_dense = true;

    cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size () << " data points." << endl;
    stringstream ss;
    ss << "cloud_cluster_" << j << ".pcd";
    writer.write<pcl::PointXYZ> (ss.str (), *cloud_cluster, false); //*
    drawBoundingBox(cloud_cluster, vec);
    j++;
  }

  ofstream fd;
  fd.open("output.json");
  Json::Value BB;
  BB["BoundingBox"] = vec;
  Json::StyledWriter styledWriter;
  fd << styledWriter.write(BB);
  fd.close();

}



int main (int argc, char** argv){

  vector<int> xyz_file_indices = pcl::console::parse_file_extension_argument (argc, argv, ".xyz");
  if (xyz_file_indices.size () != 1)
  {
    pcl::console::print_error ("Need one input XYZ file .\n");
    return (-1);
  }

  // Load the first file
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
  // pcl::PointCloud<PointXYZ> pcl::cloud;
  if (!loadCloud (argv[xyz_file_indices[0]], cloud)) 
    return (-1);

  cerr << "Cloud before filtering: " << endl;
  cerr << *cloud << endl;
  // viewCloud("Cloud after filtering", cloud_filtered);
  // pcl::visualization::CloudViewer viewer ("Simple Cloud Viewer");
  // viewer.showCloud (cloud);
  // while (!viewer.wasStopped ()){}

  // Create the filtering object
  pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
  sor.setInputCloud (cloud);
  sor.setMeanK (50);
  sor.setStddevMulThresh (1.0);
  sor.filter (*cloud_filtered);

  cerr << "Cloud after filtering: " << endl;
  cerr << *cloud_filtered << endl;
  // viewCloud("Cloud after filtering", cloud_filtered);
  // viewer.showCloud (cloud_filtered);
  // while (!viewer.wasStopped ()){}

  // planar_segmentation(cloud_filtered);
  // viewCloud("Cloud after filtering", cloud_filtered);
  nearestNeighbor(cloud_filtered);
  return (0);
}
