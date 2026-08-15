import json
import os

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render, get_object_or_404, redirect
from django.core.files.storage import default_storage

from .models import AnalysisRun, Account, GraphEdge


def dashboard_home(request):
    run_id = request.GET.get("run")
    if run_id:
        run = AnalysisRun.objects.filter(id=run_id).first()
    else:
        run = AnalysisRun.objects.filter(status="done").order_by("-created_at").first()
    if not run:
        return render(request, "detector/no_data.html")

    flagged_qs = Account.objects.filter(run=run, is_flagged=True).order_by("-prob_illicit")

    family_filter = request.GET.get("family", "")
    type_filter = request.GET.get("type", "")
    if family_filter:
        flagged_qs = flagged_qs.filter(predicted_family=family_filter)
    if type_filter:
        flagged_qs = flagged_qs.filter(predicted_type=type_filter)

    families = (Account.objects.filter(run=run, is_flagged=True)
                .exclude(predicted_family="").values_list("predicted_family", flat=True).distinct())
    types = (Account.objects.filter(run=run, is_flagged=True)
             .exclude(predicted_type="").values_list("predicted_type", flat=True).distinct())

    total_accounts = Account.objects.filter(run=run).count()
    total_flagged = Account.objects.filter(run=run, is_flagged=True).count()

    all_runs = AnalysisRun.objects.order_by("-created_at")[:10]

    context = {
        "run": run,
        "accounts": flagged_qs[:200],
        "families": sorted(set(families)),
        "types": sorted(set(types)),
        "selected_family": family_filter,
        "selected_type": type_filter,
        "total_accounts": total_accounts,
        "total_flagged": total_flagged,
        "all_runs": all_runs,
    }
    return render(request, "detector/dashboard.html", context)


def account_detail(request, run_id, node_id):
    run = get_object_or_404(AnalysisRun, id=run_id)
    account = get_object_or_404(Account, run=run, node_id=node_id)
    attributions = account.attributions.all()
    neighbor_influences = account.neighbor_influences.all()
    return render(request, "detector/account_detail.html", {
        "run": run, "account": account,
        "attributions": attributions, "neighbor_influences": neighbor_influences,
    })


def account_graph_json(request, run_id, node_id):
    run = get_object_or_404(AnalysisRun, id=run_id)
    account = get_object_or_404(Account, run=run, node_id=node_id)

    neighbor_ids = set(account.neighbor_influences.values_list("neighbor_node_id", flat=True))
    all_ids = neighbor_ids | {node_id}

    node_lookup = {a.node_id: a for a in Account.objects.filter(run=run, node_id__in=all_ids)}

    nodes = []
    for nid in all_ids:
        acc = node_lookup.get(nid)
        nodes.append({
            "id": nid,
            "label": f"#{nid}",
            "color": "#d64545" if (acc and acc.is_flagged) else "#4a90d9" if acc else "#999999",
            "title": (f"P(illicit)={acc.prob_illicit:.3f}\n{acc.predicted_type or ''}" if acc else "خارج از نمونه"),
            "isCenter": nid == node_id,
        })

    edges_qs = GraphEdge.objects.filter(run=run).filter(
        source_node_id__in=all_ids, target_node_id__in=all_ids,
    )
    edges = [{"from": e.source_node_id, "to": e.target_node_id} for e in edges_qs]

    return JsonResponse({"nodes": nodes, "edges": edges, "center": node_id})


def flagged_network_json(request, run_id):
    run = get_object_or_404(AnalysisRun, id=run_id)
    flagged = Account.objects.filter(run=run, is_flagged=True)
    flagged_ids = set(flagged.values_list("node_id", flat=True))

    nodes = [{
        "id": a.node_id,
        "label": f"#{a.node_id}",
        "color": "#d64545" if a.actual_label == 1 else "#e8a33d",
        "title": f"P(illicit)={a.prob_illicit:.3f}\n{a.predicted_type}\n{a.predicted_family}",
    } for a in flagged]

    edges_qs = GraphEdge.objects.filter(
        run=run, source_node_id__in=flagged_ids, target_node_id__in=flagged_ids,
    )
    edges = [{"from": e.source_node_id, "to": e.target_node_id} for e in edges_qs]

    return JsonResponse({"nodes": nodes, "edges": edges})


def upload_view(request):
    if request.method == "POST" and request.FILES.get("csv_file"):
        f = request.FILES["csv_file"]
        run = AnalysisRun.objects.create(
            name=f"آپلود جدید: {f.name}", status="pending", source_file_name=f.name,
        )
        upload_dir = os.path.join(settings.BASE_DIR, "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        save_path = os.path.join(upload_dir, f"run_{run.id}_{f.name}")
        with open(save_path, "wb") as out:
            for chunk in f.chunks():
                out.write(chunk)

        from .pipeline import process_upload_async
        process_upload_async(run.id, save_path)

        return redirect("detector:run_status", run_id=run.id)

    return render(request, "detector/upload.html")


def run_status(request, run_id):
    run = get_object_or_404(AnalysisRun, id=run_id)
    return render(request, "detector/run_status.html", {"run": run})


def run_status_json(request, run_id):
    run = get_object_or_404(AnalysisRun, id=run_id)
    return JsonResponse({
        "status": run.status,
        "current_step": run.current_step,
        "progress_pct": run.progress_pct,
        "notes": run.notes,
    })
