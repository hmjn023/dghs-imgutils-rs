use dghs_imgutils_rs::hub::{download_all_models, hf_hub_download, model_list::ALL_MODELS};
use std::env;

fn print_help() {
    println!("Usage:");
    println!(
        "  cargo run -- --download-all        登録されているすべてのモデルを一括ダウンロードします"
    );
    println!("  cargo run -- --list                登録されているモデルの一覧を表示します");
    println!("  cargo run -- <repo_id> <filename> [<repo_type>]");
    println!(
        "                                     指定したモデルファイルを個別にダウンロードします"
    );
    println!("                                     repo_type: model (デフォルト) または dataset");
    println!();
    println!("Examples:");
    println!("  cargo run -- deepghs/ccip_onnx ccip-caformer-24-randaug-pruned/metrics.json");
    println!("  cargo run -- alea31415/tag_filtering blacklist_tags.txt dataset");
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_help();
        return;
    }

    match args[1].as_str() {
        "--help" | "-h" => {
            print_help();
        }
        "--list" => {
            println!("登録されているモデル一覧 (計 {} 件):", ALL_MODELS.len());
            println!("{:-<120}", "");
            println!(
                "{:<35} | {:<50} | {:<8} | 説明",
                "リポジトリ ID", "ファイル名", "種類"
            );
            println!("{:-<120}", "");
            for entry in ALL_MODELS {
                println!(
                    "{:<35} | {:<50} | {:<8} | {}",
                    entry.repo_id, entry.filename, entry.repo_type, entry.description
                );
            }
            println!("{:-<120}", "");
        }
        "--download-all" => {
            println!(
                "すべてのモデルの一括ダウンロードを開始します (計 {} 件)...",
                ALL_MODELS.len()
            );
            match download_all_models(None) {
                Ok(_) => {
                    println!("\n✓ すべてのモデルのダウンロード/検証が完了しました！");
                }
                Err(e) => {
                    eprintln!("\n✗ ダウンロード中にエラーが発生しました: {e}");
                    std::process::exit(1);
                }
            }
        }
        repo_id => {
            if args.len() < 3 {
                eprintln!("エラー: ファイル名が指定されていません。");
                print_help();
                std::process::exit(1);
            }
            let filename = &args[2];
            let repo_type = if args.len() >= 4 {
                Some(args[3].as_str())
            } else {
                None
            };

            // ALL_MODELS の定義と照らし合わせて検証・アシスト
            let matches_repo = ALL_MODELS.iter().any(|entry| entry.repo_id == repo_id);
            let matches_both = ALL_MODELS
                .iter()
                .any(|entry| entry.repo_id == repo_id && entry.filename == filename);

            if !matches_both {
                println!(
                    "⚠️ 警告: 指定されたファイル名 '{}' は登録リストにありません。",
                    filename
                );
                if matches_repo {
                    println!(
                        "リポジトリ '{}' に登録されている正しいファイル名の候補:",
                        repo_id
                    );
                    for entry in ALL_MODELS.iter().filter(|entry| entry.repo_id == repo_id) {
                        println!("  - {} ({})", entry.filename, entry.description);
                    }
                } else {
                    println!("部分一致するリポジトリの候補:");
                    let mut found_any = false;
                    for entry in ALL_MODELS.iter() {
                        if entry.repo_id.contains(repo_id) {
                            println!("  - {} (ファイル: {})", entry.repo_id, entry.filename);
                            found_any = true;
                        }
                    }
                    if !found_any {
                        println!("  (一致するリポジトリが見つかりません)");
                    }
                }
                println!();
            }

            println!(
                "個別ダウンロード中: {}/{} (種類: {})",
                repo_id,
                filename,
                repo_type.unwrap_or("model")
            );

            match hf_hub_download(repo_id, filename, repo_type, None) {
                Ok(path) => {
                    println!("✓ キャッシュ/保存先: {}", path.display());
                }
                Err(e) => {
                    eprintln!("✗ ダウンロードエラー: {e}");
                    std::process::exit(1);
                }
            }
        }
    }
}
