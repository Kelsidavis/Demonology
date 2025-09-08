function Localize()
	-- Put all locale specific string adjustments here
	ACCOUNT_CREATE_URL = "http://signup.wow-europe.com/";
	AUTH_ALREADY_ONLINE = "This character is still logged on. If this character is not logged in and you continue to experience this issue for more than 15 minutes, please contact our Technical Support Department at http://www.wow-europe.com/en/support/"; 
	AUTH_BANNED = "This account has been banned for violating the Terms of Use Agreement - http://www.wow-europe.com/en/legal. Please contact our GM department at http://www.wow-europe.com/en/support/ for more information."; 
	AUTH_BANNED_URL = "http://www.wow-europe.com/en/legal"; 
	AUTH_DB_BUSY = "This session has timed out. Please try again at a later time or check the status of our WoW realms at http://www.wow-europe.com/en/serverstatus"; 
	AUTH_DB_BUSY_URL = "http://www.wow-europe.com/en/serverstatus"; 
	AUTH_NO_TIME = "Your World of Warcraft subscription has expired. You will need to reactivate your account. To do so, please visit http://signup.wow-europe.com/ for more information."; 
	AUTH_NO_TIME_URL = "http://www.wow-europe.com/en/account/"; 
	AUTH_REJECT = "Login unavailable - Please contact Technical Support at http://www.wow-europe.com/en/support/"; 
	AUTH_SUSPENDED = "This account has been temporarily suspended for violating the Terms of Use Agreement - http://www.wow-europe.com/en/legal. Please contact our GM department at http://www.wow-europe.com/en/support/ for more information."; 
	AUTH_SUSPENDED_URL = "http://www.wow-europe.com/en/legal";
	CATEGORY_DESCRIPTION = "Realm Language";
	CATEGORY_DESCRIPTION_TEXT = "The realm language is the language used by players and Game Masters in that realm. Players should use that language when they speak in general channels.";
	CHAR_CREATE_PVP_TEAMS_VIOLATION = "You cannot have both a Horde and an Alliance character on the same PvP realm";
	CHOOSE_LOCATION = "Choose your language:";
	CHOOSE_LOCATION_DESCRIPTION = "(choose the language in which you will speak while in-game and receive customer support if needed)";
	COMMUNITY_URL = "http://www.wow-europe.com/en"; 
	GAMETYPE_PVE_TEXT = "These realms allow you focus on adventuring and fighting monsters. Other players can't attack you unless you decide to permit it by enabling yourself for PvP combat.";
	LOAD_NEW = "Recommended";
	LOGIN_BADVERSION = "Unable to validate game version. This may be caused by file corruption or the interference of another program. Please visit http://www.wow-europe.com/en/support/ for more information and possible solutions to this issue."; 
	LOGIN_BANNED = "This World of Warcraft account has been closed and is no longer available for use.  Please go to http://www.wow-europe.com/en/misc/banned.html for further information.";
	LOGIN_SUSPENDED = "This World of Warcraft account has been temporarily suspended.  Please go to http://www.wow-europe.com/en/misc/banned.html for further information.";
	LOGIN_UNKNOWN_ACCOUNT = "The information you have entered is not valid. Please check the spelling of the account name and password. If you need help in retrieving a lost or stolen password and account, see http://www.wow-europe.com for more information."; 
	PVP_PARENTHESES = "PVP";
	REALM_DESCRIPTION_TEXT = "A realm is a discrete game world that exists only for the players within it. You can interact with all the players in your realm, but not with players in other realms. You cannot move your characters between realms. Realms are differentiated by language and play style.";
	REALM_LIST_REALM_NOT_FOUND = "The game server you have chosen is currently down. Use the Change Realm button to choose another Realm. Check http://www.wow-europe.com/en/serverstatus for current server status."; 
	RESPONSE_FAILED_TO_CONNECT = "Failed to connect. Please be sure that your computer is currently connected to the internet, and that no security features on your system might be blocking traffic. See www.wow-europe.com/en/support for more information."; 
	RP_PARENTHESES = "RP";
	RPPVP_PARENTHESES = "RPPVP";
	GAMETYPE_RPPVP_TEXT = "These PvP realms have strict naming conventions and behavior rules for players interested in immersing themselves as a character in a fantasy-based world.  They also focus on player combat; you are always at risk of being attacked by opposing players except in starting areas and cities.";
	SERVER_ALERT_URL = "http://status.wow-europe.com/en/alert"; 
	TECH_SUPPORT_URL = "http://www.wow-europe.com/en/support/";
	
	--SHOW_CONTEST_AGREEMENT = 1;
end

function LocalizeFrames()
	-- Put all locale specific UI adjustments here
	WorldOfWarcraftRating:SetTexture("Interface\\Glues\\Login\\Glues-FrenchRating");
	WorldOfWarcraftRating:ClearAllPoints();
	WorldOfWarcraftRating:SetPoint("BOTTOMLEFT", "AccountLoginUI", "BOTTOMLEFT", 20, 45);
	WorldOfWarcraftRating:Show();
end
