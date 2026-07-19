#!/usr/bin/env python3
"""Streamlit UI for tennis form comparison"""

import streamlit as st
import os
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
import base64

from comparison_analyzer import create_comparison_analyzer


def get_video_download_link(video_path: str, filename: str) -> str:
    """Generate download link for video file"""
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    b64_video = base64.b64encode(video_bytes).decode()
    return f'<a href="data:video/mp4;base64,{b64_video}" download="{filename}">📹 Download {filename}</a>'


def get_csv_download_link(csv_path: str, filename: str) -> str:
    """Generate download link for CSV file"""
    with open(csv_path, "rb") as f:
        csv_bytes = f.read()
    b64_csv = base64.b64encode(csv_bytes).decode()
    return f'<a href="data:text/csv;base64,{b64_csv}" download="{filename}">📊 Download {filename}</a>'


def display_video_info(uploaded_file, label: str, texts=None):
    """Display video information"""
    if uploaded_file:
        st.write(f"**{label}**")
        col1, col2, col3 = st.columns(3)
        
        # Use default English labels if texts not provided
        file_name_label = texts.get('file_name', 'File Name') if texts else 'File Name'
        file_size_label = texts.get('file_size', 'File Size') if texts else 'File Size'
        file_type_label = texts.get('file_type', 'File Type') if texts else 'File Type'
        
        with col1:
            st.metric(file_name_label, uploaded_file.name)
        with col2:
            st.metric(file_size_label, f"{uploaded_file.size / 1024 / 1024:.1f} MB")
        with col3:
            st.metric(file_type_label, uploaded_file.type)


def comparison_ui(language="English"):
    """Main comparison UI"""
    
    # Language-specific texts
    if language == "日本語":
        texts = {
            'title': "⚖️ テニスフォーム比較分析",
            'description': "2つのテニス動画を並べて比較し、フォームの違いを分析します",
            'settings': "⚙️ 比較設定",
            'threshold': "差分閾値（度）",
            'threshold_help': "この閾値を超える角度差は赤色でハイライトされます",
            'analysis_options': "📊 分析オプション",
            'show_charts': "インタラクティブチャートを表示",
            'show_frame_analysis': "フレーム別分析を表示",
            'tip': "💡 **ヒント**: 最適な比較結果を得るには、似たカメラアングルの2つのテニス動画をアップロードしてください",
            'video1_header': "🎾 動画1（参照用）",
            'video1_upload': "最初のテニス動画をアップロード",
            'video1_help': "これが参照動画として使用されます",
            'video1_label': "参照動画",
            'video2_header': "🎾 動画2（比較用）",
            'video2_upload': "2番目のテニス動画をアップロード",
            'video2_help': "この動画が参照動画と比較されます",
            'video2_label': "比較動画",
            'start_analysis': "🚀 比較分析を開始",
            'analyzing': "🔄 動画を分析中... 数分かかる場合があります。",
            'extracting_video1': "📹 動画1からポーズを抽出中...",
            'extracting_video2': "📹 動画2からポーズを抽出中...",
            'comparing_poses': "⚖️ ポーズを比較中...",
            'creating_video': "🎬 比較動画を作成中...",
            'exporting_metrics': "📊 メトリクスをエクスポート中...",
            'analysis_complete': "✅ 分析完了！",
            'success_message': "🎉 比較分析が正常に完了しました！",
            'analysis_summary': "📈 分析サマリー",
            'total_frames': "総フレーム数",
            'avg_difference': "平均差分",
            'max_difference': "最大差分",
            'high_diff_instances': "高差分インスタンス",
            'comparison_video': "🎬 並列比較動画",
            'download_results': "📥 結果をダウンロード",
            'download_video': "📹 comparison_analysis.mp4をダウンロード",
            'download_csv': "📊 comparison_metrics.csvをダウンロード",
            'interactive_charts': "📊 インタラクティブ分析チャート",
            'frame_analysis': "🔍 フレーム別分析",
            'select_frame': "フレームを選択",
            'joint_angles_video1': "📐 関節角度 - 動画1",
            'joint_angles_video2': "📐 関節角度 - 動画2",
            'angle_differences': "📊 角度差分",
            'joint_specific': "🎯 関節別分析",
            'select_joint': "詳細分析する関節を選択",
            'no_valid_data': "有効なデータが見つかりません",
            'no_metrics': "比較メトリクスが生成されませんでした。動画を確認してください。",
            'analysis_failed': "分析に失敗しました: {error}",
            'upload_both': "👆 比較分析を開始するには両方の動画をアップロードしてください",
            'use_cases': "💡 使用例とヒント",
            'file_name': "ファイル名",
            'file_size': "ファイルサイズ",
            'file_type': "ファイルタイプ",
            'joint': "関節",
            'angle': "角度",
            'difference': "差分",
            'status': "ステータス",
            'high': "高",
            'normal': "通常"
        }
    else:
        texts = {
            'title': "⚖️ Tennis Form Comparison Analysis",
            'description': "Compare two tennis videos side-by-side to analyze form differences",
            'settings': "⚙️ Comparison Settings",
            'threshold': "Difference Threshold (degrees)",
            'threshold_help': "Angle differences above this threshold will be highlighted in red",
            'analysis_options': "📊 Analysis Options",
            'show_charts': "Show Interactive Charts",
            'show_frame_analysis': "Show Frame-by-Frame Analysis",
            'tip': "💡 **Tip**: Upload two tennis videos with similar camera angles for best comparison results",
            'video1_header': "🎾 Video 1 (Reference)",
            'video1_upload': "Upload first tennis video",
            'video1_help': "This will be used as the reference video",
            'video1_label': "Reference Video",
            'video2_header': "🎾 Video 2 (Comparison)",
            'video2_upload': "Upload second tennis video",
            'video2_help': "This video will be compared against the reference",
            'video2_label': "Comparison Video",
            'start_analysis': "🚀 Start Comparison Analysis",
            'analyzing': "🔄 Analyzing videos... This may take several minutes.",
            'extracting_video1': "📹 Extracting poses from Video 1...",
            'extracting_video2': "📹 Extracting poses from Video 2...",
            'comparing_poses': "⚖️ Comparing poses...",
            'creating_video': "🎬 Creating comparison video...",
            'exporting_metrics': "📊 Exporting metrics...",
            'analysis_complete': "✅ Analysis complete!",
            'success_message': "🎉 Comparison analysis completed successfully!",
            'analysis_summary': "📈 Analysis Summary",
            'total_frames': "Total Frames",
            'avg_difference': "Avg Difference",
            'max_difference': "Max Difference",
            'high_diff_instances': "High Diff Instances",
            'comparison_video': "🎬 Side-by-Side Comparison Video",
            'download_results': "📥 Download Results",
            'download_video': "📹 Download comparison_analysis.mp4",
            'download_csv': "📊 Download comparison_metrics.csv",
            'interactive_charts': "📊 Interactive Analysis Charts",
            'frame_analysis': "🔍 Frame-by-Frame Analysis",
            'select_frame': "Select Frame",
            'joint_angles_video1': "📐 Joint Angles - Video 1",
            'joint_angles_video2': "📐 Joint Angles - Video 2",
            'angle_differences': "📊 Angle Differences",
            'joint_specific': "🎯 Joint-Specific Analysis",
            'select_joint': "Select Joint for Detailed Analysis",
            'no_valid_data': "No valid data found for {joint}",
            'no_metrics': "No comparison metrics generated. Please check your videos.",
            'analysis_failed': "Analysis failed: {error}",
            'upload_both': "👆 Please upload both videos to start the comparison analysis",
            'use_cases': "💡 Use Cases and Tips",
            'file_name': "File Name",
            'file_size': "File Size",
            'file_type': "File Type",
            'joint': "Joint",
            'angle': "Angle",
            'difference': "Difference",
            'status': "Status",
            'high': "High",
            'normal': "Normal"
        }
    
    st.title(texts['title'])
    st.markdown(texts['description'])
    
    # Sidebar configuration
    with st.sidebar:
        st.header(texts['settings'])
        
        difference_threshold = st.slider(
            texts['threshold'],
            min_value=5.0,
            max_value=50.0,
            value=20.0,
            step=5.0,
            help=texts['threshold_help']
        )
        
        st.header(texts['analysis_options'])
        show_charts = st.checkbox(texts['show_charts'], value=True)
        show_frame_by_frame = st.checkbox(texts['show_frame_analysis'], value=False)
        
        st.info(texts['tip'])
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.header(texts['video1_header'])
        uploaded_file1 = st.file_uploader(
            texts['video1_upload'],
            type=['mp4', 'avi', 'mov', 'mkv'],
            key="video1",
            help=texts['video1_help']
        )
        display_video_info(uploaded_file1, texts['video1_label'], texts)
    
    with col2:
        st.header(texts['video2_header'])
        uploaded_file2 = st.file_uploader(
            texts['video2_upload'],
            type=['mp4', 'avi', 'mov', 'mkv'],
            key="video2",
            help=texts['video2_help']
        )
        display_video_info(uploaded_file2, texts['video2_label'], texts)
    
    # Analysis section
    if uploaded_file1 and uploaded_file2:
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(texts['start_analysis'], type="primary", use_container_width=True):
                
                # Save uploaded files temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp1:
                    tmp1.write(uploaded_file1.getbuffer())
                    temp_video1 = tmp1.name
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp2:
                    tmp2.write(uploaded_file2.getbuffer())
                    temp_video2 = tmp2.name
                
                try:
                    with st.spinner(texts['analyzing']):
                        # Initialize analyzer
                        analyzer = create_comparison_analyzer(difference_threshold)
                        analyzer.initialize_components()
                        
                        # Progress tracking
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Extract poses from both videos
                        status_text.text(texts['extracting_video1'])
                        progress_bar.progress(10)
                        poses1 = analyzer.extract_poses_from_video(temp_video1)
                        
                        status_text.text(texts['extracting_video2'])
                        progress_bar.progress(30)
                        poses2 = analyzer.extract_poses_from_video(temp_video2)
                        
                        # Compare poses
                        status_text.text(texts['comparing_poses'])
                        progress_bar.progress(50)
                        comparison_metrics = analyzer.compare_poses(poses1, poses2)
                        
                        # Create comparison video
                        status_text.text(texts['creating_video'])
                        progress_bar.progress(70)
                        
                        output_dir = Path("comparison_output")
                        output_dir.mkdir(exist_ok=True)
                        
                        comparison_video_path = output_dir / "comparison_analysis.mp4"
                        analyzer.create_comparison_video(
                            poses1, poses2, comparison_metrics, 
                            str(comparison_video_path), fps=30.0
                        )
                        
                        # Export metrics
                        status_text.text(texts['exporting_metrics'])
                        progress_bar.progress(90)
                        
                        comparison_csv_path = output_dir / "comparison_metrics.csv"
                        analyzer.export_comparison_metrics(comparison_metrics, str(comparison_csv_path))
                        
                        progress_bar.progress(100)
                        status_text.text(texts['analysis_complete'])
                        
                        # Cleanup
                        analyzer.cleanup()
                    
                    # Display results
                    st.success(texts['success_message'])
                    
                    # Summary statistics
                    st.header(texts['analysis_summary'])
                    
                    if comparison_metrics:
                        # Calculate summary stats
                        overall_diffs = [m.overall_difference for m in comparison_metrics]
                        total_highlights = sum(len(m.highlight_joints) for m in comparison_metrics)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(texts['total_frames'], len(comparison_metrics))
                        with col2:
                            st.metric(texts['avg_difference'], f"{np.mean(overall_diffs):.1f}°")
                        with col3:
                            st.metric(texts['max_difference'], f"{np.max(overall_diffs):.1f}°")
                        with col4:
                            st.metric(texts['high_diff_instances'], total_highlights)
                        
                        # Show comparison video
                        st.header(texts['comparison_video'])
                        
                        if os.path.exists(comparison_video_path):
                            st.video(str(comparison_video_path))
                            
                            # Download links
                            st.markdown(f"### {texts['download_results']}")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown(
                                    get_video_download_link(str(comparison_video_path), "comparison_analysis.mp4").replace("📹 Download comparison_analysis.mp4", texts['download_video']),
                                    unsafe_allow_html=True
                                )
                            
                            with col2:
                                if os.path.exists(comparison_csv_path):
                                    st.markdown(
                                        get_csv_download_link(str(comparison_csv_path), "comparison_metrics.csv").replace("📊 Download comparison_metrics.csv", texts['download_csv']),
                                        unsafe_allow_html=True
                                    )
                        
                        # Interactive charts
                        if show_charts:
                            st.header(texts['interactive_charts'])
                            fig = analyzer.create_comparison_charts(comparison_metrics)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Frame-by-frame analysis
                        if show_frame_by_frame:
                            st.header(texts['frame_analysis'])
                            
                            frame_selector = st.slider(
                                texts['select_frame'],
                                min_value=0,
                                max_value=len(comparison_metrics) - 1,
                                value=0
                            )
                            
                            selected_metric = comparison_metrics[frame_selector]
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader(texts['joint_angles_video1'])
                                angles1_df = pd.DataFrame([
                                    {texts['joint']: joint, texts['angle']: f"{angle:.1f}°" if angle is not None else "N/A"}
                                    for joint, angle in selected_metric.angles_video1.items()
                                ])
                                st.dataframe(angles1_df, use_container_width=True)
                            
                            with col2:
                                st.subheader(texts['joint_angles_video2'])
                                angles2_df = pd.DataFrame([
                                    {texts['joint']: joint, texts['angle']: f"{angle:.1f}°" if angle is not None else "N/A"}
                                    for joint, angle in selected_metric.angles_video2.items()
                                ])
                                st.dataframe(angles2_df, use_container_width=True)
                            
                            # Differences table
                            st.subheader(texts['angle_differences'])
                            diff_data = []
                            for joint, diff in selected_metric.angle_differences.items():
                                if diff is not None:
                                    highlight = "🔴" if joint in selected_metric.highlight_joints else ""
                                    diff_data.append({
                                        texts['joint']: joint,
                                        texts['difference']: f"{diff:.1f}°",
                                        texts['status']: f"{highlight} {texts['high'] if joint in selected_metric.highlight_joints else texts['normal']}"
                                    })
                            
                            diff_df = pd.DataFrame(diff_data)
                            st.dataframe(diff_df, use_container_width=True)
                        
                        # Joint-specific analysis
                        st.header(texts['joint_specific'])
                        
                        selected_joint = st.selectbox(
                            texts['select_joint'],
                            options=analyzer.key_joints,
                            index=0
                        )
                        
                        # Plot selected joint over time
                        frames = [m.frame_idx for m in comparison_metrics]
                        angles1_joint = [m.angles_video1.get(selected_joint) for m in comparison_metrics]
                        angles2_joint = [m.angles_video2.get(selected_joint) for m in comparison_metrics]
                        differences_joint = [m.angle_differences.get(selected_joint) for m in comparison_metrics]
                        
                        # Filter out None values
                        valid_frames = []
                        valid_angles1 = []
                        valid_angles2 = []
                        valid_diffs = []
                        
                        for i, (a1, a2, diff) in enumerate(zip(angles1_joint, angles2_joint, differences_joint)):
                            if a1 is not None and a2 is not None and diff is not None:
                                valid_frames.append(frames[i])
                                valid_angles1.append(a1)
                                valid_angles2.append(a2)
                                valid_diffs.append(diff)
                        
                        if valid_frames:
                            fig_joint = go.Figure()
                            
                            fig_joint.add_trace(go.Scatter(
                                x=valid_frames, y=valid_angles1,
                                name="Video 1", line=dict(color="green", width=2)
                            ))
                            
                            fig_joint.add_trace(go.Scatter(
                                x=valid_frames, y=valid_angles2,
                                name="Video 2", line=dict(color="blue", width=2)
                            ))
                            
                            fig_joint.add_trace(go.Scatter(
                                x=valid_frames, y=valid_diffs,
                                name="Difference", line=dict(color="red", width=2),
                                yaxis="y2"
                            ))
                            
                            fig_joint.update_layout(
                                title=f"{selected_joint.replace('_', ' ').title()} Analysis",
                                xaxis_title="Frame",
                                yaxis_title="Angle (degrees)",
                                yaxis2=dict(title="Difference (degrees)", overlaying="y", side="right"),
                                height=400
                            )
                            
                            st.plotly_chart(fig_joint, use_container_width=True)
                        
                        else:
                            st.warning(texts['no_valid_data'].format(joint=selected_joint))
                    
                    else:
                        st.error(texts['no_metrics'])
                
                except Exception as e:
                    st.error(texts['analysis_failed'].format(error=str(e)))
                    import traceback
                    st.code(traceback.format_exc())
                
                finally:
                    # Cleanup temporary files
                    try:
                        os.unlink(temp_video1)
                        os.unlink(temp_video2)
                    except:
                        pass
    
    else:
        st.info(texts['upload_both'])
        
        # Show example use cases
        with st.expander(texts['use_cases']):
            st.markdown("""
            **Perfect for comparing:**
            - Professional vs. amateur tennis techniques
            - Before and after coaching sessions
            - Different playing styles or techniques
            - Left-handed vs. right-handed players (with mirroring)
            
            **Best results when:**
            - Videos have similar camera angles and distances
            - Players are clearly visible throughout the video
            - Similar lighting conditions
            - Similar video frame rates
            
            **The analysis will show:**
            - Side-by-side video comparison with pose overlays
            - Joint angles for both players
            - Highlighted differences above the threshold
            - Frame-by-frame numerical analysis
            - Interactive charts and trends
            """)


if __name__ == "__main__":
    comparison_ui("English")