from django.urls import path
from . import views

app_name = "detector"

urlpatterns = [
    path("", views.dashboard_home, name="dashboard_home"),
    path("run/<int:run_id>/account/<int:node_id>/", views.account_detail, name="account_detail"),
    path("run/<int:run_id>/account/<int:node_id>/graph.json", views.account_graph_json, name="account_graph_json"),
    path("run/<int:run_id>/network.json", views.flagged_network_json, name="flagged_network_json"),
    path("upload/", views.upload_view, name="upload"),
    path("run/<int:run_id>/status/", views.run_status, name="run_status"),
    path("run/<int:run_id>/status.json", views.run_status_json, name="run_status_json"),
]
