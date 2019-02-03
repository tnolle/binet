#  Copyright 2018 Timo Nolle
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#  ==============================================================================

user_names = ['Roy', 'Earl', 'James', 'Charles', 'Ryan', 'Marilyn', 'Emily', 'Craig', 'Howard', 'Amanda', 'Johnny',
              'Brian', 'Jack', 'Paul', 'Joe', 'Ronald', 'Donald', 'Anna', 'Steve', 'Lisa', 'Gema', 'Doretta', 'Hannah',
              'Maryellen', 'Pam', 'Sherell', 'Micheline', 'Shandi', 'Hugo', 'Jamika', 'Brant', 'Rossana', 'Della',
              'Velda', 'Hoyt', 'Tiffiny', 'Frances', 'Alpha', 'Jimmy', 'Junior', 'Issac', 'Evelin', 'Deloras', 'Hassie',
              'Josef', 'Clayton', 'Sandra', 'Rossie', 'Vickie', 'Lourdes', 'Jin', 'Sigrid', 'Elisha', 'Sherlene',
              'Lucy', 'Chan', 'Lannie', 'Alyce', 'Melany', 'Wilton', 'Seth', 'Sonia', 'Iluminada', 'Michaele', 'Ling',
              'Keven', 'Roseanne', 'Sharee', 'Carmella', 'Grayce']

working_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
weekend_days = ['Saturday', 'Sunday']
week_days = working_days + weekend_days

company_names = ['Openlane', 'Yearin', 'Goodsilron', 'Condax', 'Opentech', 'Golddex', 'year-job', 'Isdom', 'Gogozoom',
                 'Y-corporation', 'Nam-zim', 'Donquadtech', 'Warephase', 'Donware', 'Faxquote', 'Sunnamplex',
                 'Lexiqvolax', 'Sumace', 'Treequote', 'Iselectrics', 'Zencorporation', 'Plusstrip', 'dambase',
                 'Toughzap', 'Codehow', 'Zotware', 'Statholdings', 'Conecom', 'Zathunicon', 'Labdrill', 'Ron-tech',
                 'Green-Plus', 'Groovestreet', 'Zoomit', 'Bioplex', 'Zumgoity', 'Scotfind', 'Dalttechnology',
                 'Kinnamplus', 'Konex', 'Stanredtax', 'Cancity', 'Finhigh', 'Kan-code', 'Blackzim', 'Dontechi',
                 'Xx-zobam', 'Fasehatice', 'Hatfan', 'Streethex', 'Inity', 'Konmatfix', 'Bioholding', 'Hottechi',
                 'Ganjaflex', 'Betatech', 'Domzoom', 'Ontomedia', 'Newex', 'Betasoloin', 'Mathtouch', 'Rantouch',
                 'Silis', 'Plussunin', 'Plexzap', 'Finjob', 'Xx-holding', 'Scottech', 'Funholding', 'Sonron',
                 'Singletechno', 'Rangreen', 'J-Texon', 'Rundofase', 'Doncon']

countries = ['Afghanistan', 'Aland Islands', 'Albania', 'Algeria', 'American Samoa', 'Andorra', 'Angola', 'Anguilla',
             'Antarctica', 'Antigua And Barbuda', 'Argentina', 'Armenia', 'Aruba', 'Australia', 'Austria', 'Azerbaijan',
             'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin', 'Bermuda',
             'Bhutan', 'Bolivia', 'Bosnia And Herzegovina', 'Botswana', 'Bouvet Island', 'Brazil',
             'British Indian Ocean Territory', 'Brunei Darussalam', 'Bulgaria', 'Burkina Faso', 'Burundi', 'Cambodia',
             'Cameroon', 'Canada', 'Cape Verde', 'Cayman Islands', 'Central African Republic', 'Chad', 'Chile', 'China',
             'Christmas Island', 'Cocos (Keeling) Islands', 'Colombia', 'Comoros', 'Congo',
             'Congo, Democratic Republic', 'Cook Islands', 'Costa Rica', 'Cote D\'Ivoire', 'Croatia', 'Cuba', 'Cyprus',
             'Czech Republic', 'Denmark', 'Djibouti', 'Dominica', 'Dominican Republic', 'Ecuador', 'Egypt',
             'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Estonia', 'Ethiopia', 'Falkland Islands (Malvinas)',
             'Faroe Islands', 'Fiji', 'Finland', 'France', 'French Guiana', 'French Polynesia',
             'French Southern Territories', 'Gabon', 'Gambia', 'Georgia', 'Germany', 'Ghana', 'Gibraltar', 'Greece',
             'Greenland', 'Grenada', 'Guadeloupe', 'Guam', 'Guatemala', 'Guernsey', 'Guinea', 'Guinea-Bissau', 'Guyana',
             'Haiti', 'Heard Island & Mcdonald Islands', 'Holy See (Vatican City State)', 'Honduras', 'Hong Kong',
             'Hungary', 'Iceland', 'India', 'Indonesia', 'Iran, Islamic Republic Of', 'Iraq', 'Ireland', 'Isle Of Man',
             'Israel', 'Italy', 'Jamaica', 'Japan', 'Jersey', 'Jordan', 'Kazakhstan', 'Kenya', 'Kiribati', 'Korea',
             'Kuwait', 'Kyrgyzstan', 'Lao People\'s Democratic Republic', 'Latvia', 'Lebanon', 'Lesotho', 'Liberia',
             'Libyan Arab Jamahiriya', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Macao', 'Macedonia', 'Madagascar',
             'Malawi', 'Malaysia', 'Maldives', 'Mali', 'Malta', 'Marshall Islands', 'Martinique', 'Mauritania',
             'Mauritius', 'Mayotte', 'Mexico', 'Micronesia, Federated States Of', 'Moldova', 'Monaco', 'Mongolia',
             'Montenegro', 'Montserrat', 'Morocco', 'Mozambique', 'Myanmar', 'Namibia', 'Nauru', 'Nepal', 'Netherlands',
             'Netherlands Antilles', 'New Caledonia', 'New Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'Niue',
             'Norfolk Island', 'Northern Mariana Islands', 'Norway', 'Oman', 'Pakistan', 'Palau',
             'Palestinian Territory, Occupied', 'Panama', 'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines',
             'Pitcairn', 'Poland', 'Portugal', 'Puerto Rico', 'Qatar', 'Reunion', 'Romania', 'Russian Federation',
             'Rwanda', 'Saint Barthelemy', 'Saint Helena', 'Saint Kitts And Nevis', 'Saint Lucia', 'Saint Martin',
             'Saint Pierre And Miquelon', 'Saint Vincent And Grenadines', 'Samoa', 'San Marino',
             'Sao Tome And Principe', 'Saudi Arabia', 'Senegal', 'Serbia', 'Seychelles', 'Sierra Leone', 'Singapore',
             'Slovakia', 'Slovenia', 'Solomon Islands', 'Somalia', 'South Africa', 'South Georgia And Sandwich Isl.',
             'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Svalbard And Jan Mayen', 'Swaziland', 'Sweden', 'Switzerland',
             'Syrian Arab Republic', 'Taiwan', 'Tajikistan', 'Tanzania', 'Thailand', 'Timor-Leste', 'Togo', 'Tokelau',
             'Tonga', 'Trinidad And Tobago', 'Tunisia', 'Turkey', 'Turkmenistan', 'Turks And Caicos Islands', 'Tuvalu',
             'Uganda', 'Ukraine', 'United Arab Emirates', 'United Kingdom', 'United States',
             'United States Outlying Islands', 'Uruguay', 'Uzbekistan', 'Vanuatu', 'Venezuela', 'Viet Nam',
             'Virgin Islands, British', 'Virgin Islands, U.S.', 'Wallis And Futuna', 'Western Sahara', 'Yemen',
             'Zambia', 'Zimbabwe']

countries_iso = ['AF', 'AX', 'AL', 'DZ', 'AS', 'AD', 'AO', 'AI', 'AQ', 'AG', 'AR', 'AM', 'AW', 'AU', 'AT', 'AZ', 'BS',
                 'BH', 'BD', 'BB', 'BY', 'BE', 'BZ', 'BJ', 'BM', 'BT', 'BO', 'BA', 'BW', 'BV', 'BR', 'IO', 'BN', 'BG',
                 'BF', 'BI', 'KH', 'CM', 'CA', 'CV', 'KY', 'CF', 'TD', 'CL', 'CN', 'CX', 'CC', 'CO', 'KM', 'CG', 'CD',
                 'CK', 'CR', 'CI', 'HR', 'CU', 'CY', 'CZ', 'DK', 'DJ', 'DM', 'DO', 'EC', 'EG', 'SV', 'GQ', 'ER', 'EE',
                 'ET', 'FK', 'FO', 'FJ', 'FI', 'FR', 'GF', 'PF', 'TF', 'GA', 'GM', 'GE', 'DE', 'GH', 'GI', 'GR', 'GL',
                 'GD', 'GP', 'GU', 'GT', 'GG', 'GN', 'GW', 'GY', 'HT', 'HM', 'VA', 'HN', 'HK', 'HU', 'IS', 'IN', 'ID',
                 'IR', 'IQ', 'IE', 'IM', 'IL', 'IT', 'JM', 'JP', 'JE', 'JO', 'KZ', 'KE', 'KI', 'KR', 'KW', 'KG', 'LA',
                 'LV', 'LB', 'LS', 'LR', 'LY', 'LI', 'LT', 'LU', 'MO', 'MK', 'MG', 'MW', 'MY', 'MV', 'ML', 'MT', 'MH',
                 'MQ', 'MR', 'MU', 'YT', 'MX', 'FM', 'MD', 'MC', 'MN', 'ME', 'MS', 'MA', 'MZ', 'MM', 'NA', 'NR', 'NP',
                 'NL', 'AN', 'NC', 'NZ', 'NI', 'NE', 'NG', 'NU', 'NF', 'MP', 'NO', 'OM', 'PK', 'PW', 'PS', 'PA', 'PG',
                 'PY', 'PE', 'PH', 'PN', 'PL', 'PT', 'PR', 'QA', 'RE', 'RO', 'RU', 'RW', 'BL', 'SH', 'KN', 'LC', 'MF',
                 'PM', 'VC', 'WS', 'SM', 'ST', 'SA', 'SN', 'RS', 'SC', 'SL', 'SG', 'SK', 'SI', 'SB', 'SO', 'ZA', 'GS',
                 'ES', 'LK', 'SD', 'SR', 'SJ', 'SZ', 'SE', 'CH', 'SY', 'TW', 'TJ', 'TZ', 'TH', 'TL', 'TG', 'TK', 'TO',
                 'TT', 'TN', 'TR', 'TM', 'TC', 'TV', 'UG', 'UA', 'AE', 'GB', 'US', 'UM', 'UY', 'UZ', 'VU', 'VE', 'VN',
                 'VG', 'VI', 'WF', 'EH', 'YE', 'ZM', 'ZW']
