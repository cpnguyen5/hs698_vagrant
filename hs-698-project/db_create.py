from api import db, app
# from api.models import Report, Puf, Cancer
import project

engine = db.engine

# Remove spontaneous quoting of column name
db.engine.dialect.identifier_preparer.initial_quote = ''
db.engine.dialect.identifier_preparer.final_quote = ''

#Create database and tablesfab vagrant bo
db.create_all()
print "Table(s) schema created, inserting data..."

# Insert Data -- Bulk insert of DataFrame
## Insert Report CSV -- Report Table
df_report = project.readCSV()
for elem in df_report[1:]:
    elem.to_sql('report', engine, if_exists='append', index=False)
    db.session.commit()
    print count
## Insert PUF CSV -- Puf Table
df_puf = project.readPUF()
for elem in df_puf:
    elem.to_sql('puf', engine, if_exists='append', index=False)
    db.session.commit()
# Insert BCHC CSV -- Cancer Table
df_can = project.readBCH()
df_can.to_sql('cancer', engine, if_exists='append', index=False)



#Commit changes
db.session.commit()
db.session.close()  # close session
print "Data insert successful...database initialization complete"
