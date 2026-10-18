# run.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

from app import create_app, db

if __name__ == '__main__':
    print("🚀 Starting CricketAI Application...")
    
    # Create the Flask application
    config_name = os.getenv('FLASK_CONFIG') or 'default'
    app = create_app(config_name)
    
    # Create necessary directories
    directories = [
        'static/uploads',
        'static/css', 
        'static/js',
        'app/data/images',
        'app/data/encodings'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Initialize database and load data within app context
    with app.app_context():
        try:
            # Create all database tables
            db.create_all()
            print("✅ Database tables created")
            
            # Check if we have any users, if not create a sample admin user
            from app.models import User, PlayerData
            
            if User.query.count() == 0:
                admin_user = User(
                    username='admin',
                    email='admin@cricketai.com',
                    first_name='Admin',
                    last_name='User'
                )
                admin_user.set_password('admin123')
                db.session.add(admin_user)
                db.session.commit()
                print("✅ Sample admin user created (email: admin@cricketai.com, password: admin123)")
            else:
                print("✅ Users already exist in database")
            
            # Load player data from JSON file
            import json
            player_data_file = 'player_data.json'
            if os.path.exists(player_data_file):
                try:
                    # Check if player data already exists
                    if PlayerData.query.count() == 0:
                        with open(player_data_file, 'r', encoding='utf-8') as f:
                            players = json.load(f)
                        
                        added_count = 0
                        for player in players:
                            if player.get('name'):  # Only add players with names
                                player_record = PlayerData(
                                    name=player['name'],
                                    role=player.get('role', ''),
                                    country=player.get('country', ''),
                                    dob=player.get('dob', ''),
                                    batting_style=player.get('batting_style', ''),
                                    bowling_style=player.get('bowling_style', ''),
                                    stats=json.dumps(player.get('stats', {})),
                                    achievements=json.dumps(player.get('achievements', []))
                                )
                                db.session.add(player_record)
                                added_count += 1
                        
                        db.session.commit()
                        print(f"✅ Loaded {added_count} players into database")
                    else:
                        print("✅ Player data already exists in database")
                except Exception as e:
                    print(f"⚠️  Warning: Could not load player data: {e}")
            else:
                print(f"⚠️  Warning: Player data file '{player_data_file}' not found")
            
        except Exception as e:
            print(f"❌ Error initializing database: {e}")
    
    print(f"🌐 Application running at: http://127.0.0.1:5000")
    print(f"🔧 Debug mode: {app.config['DEBUG']}")
    print(f"📁 Upload folder: {app.config['UPLOAD_FOLDER']}")
    print("=" * 50)
    
    # Run the application
    try:
        app.run(
            host='127.0.0.1',
            port=5000,
            debug=app.config['DEBUG'],
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")
